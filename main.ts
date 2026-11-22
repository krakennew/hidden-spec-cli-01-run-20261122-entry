// main.ts is mend, the CLI SPEC.md specifies: one command-line tool that applies the edits AI models write,
// in four formats, to a tree of files. Node 22, the standard library alone; run it as `node main.ts [options] [PATCH]`.
//
// mend reads a patch -- a unified diff, SEARCH/REPLACE blocks, a V4A patch, or JSON tool calls -- applies its edits
// to the tree in the working directory, and writes every change or none: a failed edit reports its code and path
// and nothing is written. Lines are matched at three tiers -- exact, trailing whitespace ignored, then leading
// and trailing whitespace ignored -- so a model's drifted whitespace still lands where it meant. A context line
// is written as the file's own line, never the patch's wording, and at the loosest tier the new lines are
// re-indented from the old block's indent to the file's.
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and then this file hands over to
// runner.ts (the last lines below): keep that hand-over and runner.ts as they are, and write the CLI.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** An exit-2 answer: one error line, nothing written. */
class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}

/** One edit that failed: its code and the path the failure concerns. */
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const blank = (s: string): boolean => s.trim() === '';

/** A patch's lines: split at \n, one trailing \r removed, no empty piece after a final \n. */
function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

/** A path is safe when it is not empty, does not start with /, holds no backslash, and none of its
 *  /-separated segments is empty, . or .. . */
function safe(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

const leadWs = (s: string): string => /^[ \t]*/.exec(s)?.[0] ?? '';
const onlyWs = (s: string): boolean => s === '' || /^[ \t]+$/.test(s);

/** A file's lines, its line break, and whether it ends with one. */
interface FileState {
  lines: string[];
  br: string;
  final: boolean;
}

/** Split a file's text: the break is CRLF when its first line feed follows a carriage return, else LF; the
 *  empty piece after a final break is not a line; an empty file has no lines and no final break. */
function parseFile(text: string): FileState {
  if (text === '') return { lines: [], br: '\n', final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const final = text.endsWith(br);
  const lines = text.split(br);
  if (final) lines.pop();
  return { lines, br, final };
}

/** Join a file's lines back with its own break, and a final one exactly when it ended with one. */
function serializeFile(f: FileState): string {
  if (f.lines.length === 0) return '';
  return f.lines.join(f.br) + (f.final ? f.br : '');
}

/** The tree as the edits leave it: changed paths in `now` (null: deleted), the rest read from disk. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }
}

type Edit = (tree: Tree) => void;

/** A move a successful move edit recorded, in patch order, for the output lines. */
interface Move {
  old: string;
  next: string;
}

/** A line as a tier reads it: exact, trailing spaces and tabs off, or both ends off. */
function tierLine(s: string, tier: number): string {
  if (tier === 0) return s;
  const t = s.replace(/[ \t]+$/, '');
  return tier === 1 ? t : t.replace(/^[ \t]+/, '');
}

/** The positions where old lines occur in the file's lines at a tier, at or after `from`. */
function occurrences(fileLines: string[], oldLines: string[], tier: number, from = 0): number[] {
  const out: number[] = [];
  if (oldLines.length === 0) return out;
  for (let i = from; i + oldLines.length <= fileLines.length; i++) {
    let ok = true;
    for (let j = 0; j < oldLines.length; j++) {
      if (tierLine(fileLines[i + j]!, tier) !== tierLine(oldLines[j]!, tier)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

/** The first tier at which old lines occur, with its positions, or null when no tier has one. With --strict
 *  only tier 0 is tried. Positions are at or after `from`, and only end at the file's last line when `eof`. */
function matchAt(fileLines: string[], oldLines: string[], strict: boolean, from = 0, eof = false): { tier: number; positions: number[] } | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    let positions = occurrences(fileLines, oldLines, t, from);
    if (eof) positions = positions.filter((p) => p + oldLines.length === fileLines.length);
    if (positions.length > 0) return { tier: t, positions };
  }
  return null;
}

/** One new line: its text, and the index into the old lines when it is a context line (one that is both old
 *  and new), so it is written as the file's own matched line. */
interface NewLine {
  text: string;
  ctx: number | null;
}

/** Render new lines against the matched occurrence: a context line is the file's own matched line, never the
 *  patch's wording. At tier 2 every other line is re-indented: the leading spaces and tabs of the file's first
 *  matched line replace those of the old block's first line, where the new line starts with them. At tiers 0
 *  and 1 the new lines are written as they are. */
function renderNew(fileLines: string[], pos: number, oldLines: string[], nlines: NewLine[], tier: number): string[] {
  if (oldLines.length === 0) return nlines.map((n) => n.text);
  const f = leadWs(fileLines[pos] ?? '');
  const p = leadWs(oldLines[0] ?? '');
  return nlines.map((n) => {
    if (n.ctx !== null) return fileLines[pos + n.ctx]!;
    if (tier !== 2) return n.text;
    if (onlyWs(n.text)) return n.text;
    return n.text.startsWith(p) ? f + n.text.slice(p.length) : n.text;
  });
}

/** Detect the patch's format: the first of V4A, JSON, SEARCH/REPLACE, unified that holds, else null. */
function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

/** One JSON edit object (SPEC.md section 9), or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const exists = (tree: Tree): string => {
    if (!safe(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const br = nl > 0 && before[nl - 1] === '\r' ? '\r\n' : '\n';
      const lines = before === '' ? [] : before.split(br);
      const final = before !== '' && before.endsWith(br);
      if (final) lines.pop();
      if (line < 0 || line > lines.length) throw new Fail('no-match', p);
      lines.splice(line, 0, ...add);
      tree.now.set(p, lines.length === 0 ? '' : lines.join(br) + (final ? br : ''));
    };
  }
  return null;
}

/** A JSON patch: one array of edit objects, or JSON Lines with one edit object a line. */
function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (blank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

/** One body line of a hunk: context, old, new, a backslash line, or an empty (context) line. */
interface BodyLine {
  kind: ' ' | '-' | '+' | '\\' | '';
  text: string;
}

/** One hunk: its old start from the header, its body, and its old and new lines in body order. */
interface Hunk {
  a: number;
  body: BodyLine[];
  oLines: string[];
  nLines: NewLine[];
}

/** Whether a backslash line follows the hunk's last context or + line: only that one counts, never one after
 *  a minus line. */
function backslashAfterLastCtxPlus(h: Hunk): boolean {
  let last = -1;
  for (let j = 0; j < h.body.length; j++) {
    const k = h.body[j]!.kind;
    if (k === ' ' || k === '+') last = j;
  }
  if (last < 0) return false;
  return h.body[last + 1]?.kind === '\\';
}

/** A hunk's body, from the line after its header: lines that start with a space, -, + or backslash, or are
 *  empty. It ends at the first other line, at a diff --git line, at a --- line followed by a +++ line, or at
 *  the end of the patch; empty lines at the end of a body are not part of it. */
function readHunk(lines: string[], at: number, a: number): { hunk: Hunk; next: number } {
  const body: BodyLine[] = [];
  let i = at + 1;
  while (i < lines.length) {
    const l = lines[i]!;
    if (l.startsWith('diff --git ')) break;
    if (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) break;
    if (l === '') {
      body.push({ kind: '', text: '' });
      i++;
      continue;
    }
    const c = l[0]!;
    if (c === ' ' || c === '-' || c === '+' || c === '\\') {
      body.push({ kind: c as BodyLine['kind'], text: l.slice(1) });
      i++;
      continue;
    }
    break;
  }
  while (body.length > 0 && body[body.length - 1]!.kind === '') body.pop();
  const oLines: string[] = [];
  const nLines: NewLine[] = [];
  for (const b of body) {
    if (b.kind === ' ' || b.kind === '') {
      // An empty line is a context line whose text is empty.
      oLines.push(b.text);
      nLines.push({ text: b.text, ctx: oLines.length - 1 });
    } else if (b.kind === '-') {
      oLines.push(b.text);
    } else if (b.kind === '+') {
      nLines.push({ text: b.text, ctx: null });
    }
  }
  return { hunk: { a, body, oLines, nLines }, next: i };
}

/** A unified section: its paths as the header names them (null: /dev/null, undefined: unnamed), its file-mode
 *  flags, and its hunks. */
interface Section {
  pairOld: string | null | undefined;
  pairNew: string | null | undefined;
  renOld: string | undefined;
  renNew: string | undefined;
  gitOld: string | undefined;
  gitNew: string | undefined;
  newFile: boolean;
  deletedFile: boolean;
  hunks: Hunk[];
}

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

/** A diff --git line's two paths, or a parse fault naming its line. */
function gitDiffPaths(line: string, lineNo: number): { old: string; next: string } {
  let m = /^diff --git a\/(.+?) b\/(.+)$/.exec(line);
  if (m) return { old: m[1]!, next: m[2]! };
  m = /^diff --git "a\/(.*)" "b\/(.*)"$/.exec(line);
  if (m) return { old: m[1]!, next: m[2]! };
  throw new Exit2(`error: parse ${lineNo}`);
}

/** A --- or +++ line's path: the text after the four characters, up to the first tab, surrounding spaces
 *  removed; /dev/null means no file, a leading a/ or b/ is removed from either. */
function diffPath(line: string): string | null {
  let p = line.slice(4);
  const tab = p.indexOf('\t');
  if (tab >= 0) p = p.slice(0, tab);
  p = p.trim();
  if (p === '/dev/null') return null;
  if (p.startsWith('a/') || p.startsWith('b/')) p = p.slice(2);
  return p;
}

/** Apply a section's hunks in order to a file's lines, each to the result of the ones before it. A hunk takes
 *  the occurrence nearest to its old start plus the running delta of new-minus-old lengths; an empty old block
 *  inserts after the first header-start-plus-delta lines. A hunk whose matched lines run to the end of the file
 *  leaves a final line break unless a backslash line follows its last context or + line. */
function applyHunks(state: FileState, hunks: Hunk[], strict: boolean, path: string): FileState {
  let d = 0;
  for (const h of hunks) {
    const oldLen = state.lines.length;
    if (h.oLines.length > 0) {
      const m = matchAt(state.lines, h.oLines, strict);
      if (m === null) throw new Fail('no-match', path);
      const target = h.a - 1 + d;
      let pos = m.positions[0]!;
      for (const p of m.positions) {
        if (Math.abs(p - target) < Math.abs(pos - target)) pos = p;
      }
      const rendered = renderNew(state.lines, pos, h.oLines, h.nLines, m.tier);
      const runsToEnd = pos + h.oLines.length === oldLen;
      state.lines.splice(pos, h.oLines.length, ...rendered);
      if (runsToEnd) state.final = !backslashAfterLastCtxPlus(h);
      d += h.nLines.length - h.oLines.length;
    } else {
      const insertAt = Math.min(h.a + d, oldLen);
      state.lines.splice(insertAt, 0, ...h.nLines.map((n) => n.text));
      if (insertAt === oldLen) state.final = !backslashAfterLastCtxPlus(h);
      d += h.nLines.length;
    }
  }
  return state;
}

/** A unified patch's sections, each one edit. A section starts at a diff --git line or at a --- line followed
 *  by a +++ line that is not already part of a section's header; outside bodies, lines that are not part of a
 *  section header are ignored. */
function unifiedEdits(lines: string[], strict: boolean, moves: Move[]): Edit[] {
  const sections: Section[] = [];
  let cur: Section | null = null;
  let inHeader = false;
  const startSection = (gitOld: string | undefined, gitNew: string | undefined): void => {
    cur = { pairOld: undefined, pairNew: undefined, renOld: undefined, renNew: undefined, gitOld, gitNew, newFile: false, deletedFile: false, hunks: [] };
    sections.push(cur);
    inHeader = true;
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (line.startsWith('diff --git ')) {
      const p = gitDiffPaths(line, i + 1);
      startSection(p.old, p.next);
      i++;
      continue;
    }
    if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      if (cur === null || !inHeader) startSection(undefined, undefined);
      if (cur!.pairOld === undefined) {
        cur!.pairOld = diffPath(line);
        cur!.pairNew = diffPath(lines[i + 1]!);
      }
      i += 2;
      continue;
    }
    if (cur !== null && inHeader) {
      if (line.startsWith('--- ')) throw new Exit2(`error: parse ${i + 2}`);
      if (line.startsWith('rename from ')) {
        if (cur.renOld === undefined) cur.renOld = line.slice(12).trim();
        i++;
        continue;
      }
      if (line.startsWith('rename to ')) {
        if (cur.renNew === undefined) cur.renNew = line.slice(10).trim();
        i++;
        continue;
      }
      if (line.startsWith('new file mode')) {
        cur.newFile = true;
        i++;
        continue;
      }
      if (line.startsWith('deleted file mode')) {
        cur.deletedFile = true;
        i++;
        continue;
      }
      if (line.startsWith('@@')) {
        const m = HUNK_RE.exec(line);
        if (m === null) throw new Exit2(`error: parse ${i + 1}`);
        const r = readHunk(lines, i, Number(m[1]));
        cur.hunks.push(r.hunk);
        inHeader = false;
        i = r.next;
        continue;
      }
      i++;
      continue;
    }
    if (cur !== null) {
      if (line.startsWith('@@')) {
        const m = HUNK_RE.exec(line);
        if (m === null) throw new Exit2(`error: parse ${i + 1}`);
        const r = readHunk(lines, i, Number(m[1]));
        cur.hunks.push(r.hunk);
        i = r.next;
        continue;
      }
      i++;
      continue;
    }
    if (line.startsWith('@@')) throw new Exit2(`error: parse ${i + 1}`);
    i++;
  }
  return sections.map((sec) => {
    const oldP = sec.pairOld !== undefined ? sec.pairOld : sec.renOld !== undefined ? sec.renOld : sec.gitOld;
    const newP = sec.pairNew !== undefined ? sec.pairNew : sec.renNew !== undefined ? sec.renNew : sec.gitNew;
    return (tree: Tree) => {
      if (oldP !== null && oldP !== undefined && !safe(oldP)) throw new Fail('unsafe', oldP);
      if (newP !== null && newP !== undefined && !safe(newP)) throw new Fail('unsafe', newP);
      if (oldP === null || sec.newFile) {
        const p = newP;
        if (p === null || p === undefined) throw new Fail('unsafe', '');
        if (tree.get(p) !== null) throw new Fail('exists', p);
        const plus: string[] = [];
        for (const h of sec.hunks) for (const n of h.nLines) if (n.ctx === null) plus.push(n.text);
        let backslash = false;
        for (let hi = sec.hunks.length - 1; hi >= 0; hi--) {
          const body = sec.hunks[hi]!.body;
          let lastPlus = -1;
          for (let j = 0; j < body.length; j++) if (body[j]!.kind === '+') lastPlus = j;
          if (lastPlus >= 0) {
            backslash = body[lastPlus + 1]?.kind === '\\';
            break;
          }
        }
        tree.now.set(p, plus.length === 0 ? '' : plus.join('\n') + (backslash ? '' : '\n'));
        return;
      }
      if (newP === null || sec.deletedFile) {
        const p = oldP!;
        if (tree.get(p) === null) throw new Fail('missing', p);
        tree.now.set(p, null);
        return;
      }
      if (oldP !== newP) {
        const before = tree.get(oldP!);
        if (before === null) throw new Fail('missing', oldP!);
        if (tree.get(newP!) !== null) throw new Fail('exists', newP!);
        const state = applyHunks(parseFile(before), sec.hunks, strict, oldP!);
        tree.now.set(newP!, serializeFile(state));
        tree.now.set(oldP!, null);
        moves.push({ old: oldP!, next: newP! });
        return;
      }
      const before = tree.get(oldP!);
      if (before === null) throw new Fail('missing', oldP!);
      const state = applyHunks(parseFile(before), sec.hunks, strict, oldP!);
      tree.now.set(oldP!, serializeFile(state));
    };
  });
}

/** A SEARCH/REPLACE patch's blocks, each one edit. The path is the nearest line before the SEARCH marker, after
 *  the previous block's REPLACE marker, that is not blank and does not start with three backticks; a block with
 *  no such line takes the previous block's path. Lines outside blocks are ignored. */
function srEdits(lines: string[], strict: boolean, moves: Move[]): Edit[] {
  void moves;
  const edits: Edit[] = [];
  let prevPath: string | null = null;
  let searchFrom = 0;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchIdx = i;
    let path: string | null = null;
    for (let j = searchIdx - 1; j >= searchFrom; j--) {
      const l = lines[j]!;
      if (blank(l)) continue;
      if (l.trim().startsWith('```')) continue;
      path = l.trim();
      break;
    }
    if (path === null) path = prevPath;
    if (path === null) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let sep = -1;
    for (let j = searchIdx + 1; j < lines.length; j++) {
      const t = lines[j]!.trimEnd();
      if (t === '=======') {
        sep = j;
        break;
      }
      if (t === '>>>>>>> REPLACE') break;
    }
    if (sep < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let rep = -1;
    for (let j = sep + 1; j < lines.length; j++) {
      if (lines[j]!.trimEnd() === '>>>>>>> REPLACE') {
        rep = j;
        break;
      }
    }
    if (rep < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    const oldLines = lines.slice(searchIdx + 1, sep);
    const newLines = lines.slice(sep + 1, rep).map((t) => ({ text: t, ctx: null as number | null }));
    const blockPath = path;
    edits.push((tree) => {
      if (!safe(blockPath)) throw new Fail('unsafe', blockPath);
      if (oldLines.length === 0) {
        if (tree.get(blockPath) !== null) throw new Fail('exists', blockPath);
        tree.now.set(blockPath, newLines.length === 0 ? '' : newLines.map((n) => n.text).join('\n') + '\n');
        return;
      }
      const before = tree.get(blockPath);
      if (before === null) throw new Fail('missing', blockPath);
      const state = parseFile(before);
      const m = matchAt(state.lines, oldLines, strict);
      if (m === null) throw new Fail('no-match', blockPath);
      if (m.positions.length > 1) throw new Fail('ambiguous', blockPath);
      const pos = m.positions[0]!;
      const rendered = renderNew(state.lines, pos, oldLines, newLines, m.tier);
      state.lines.splice(pos, oldLines.length, ...rendered);
      tree.now.set(blockPath, serializeFile(state));
    });
    prevPath = path;
    searchFrom = rep + 1;
    i = rep + 1;
  }
  return edits;
}

/** One V4A chunk: its anchor (null: none), whether it is anchored to the end of the file, and its old and new
 *  lines in body order. */
interface VChunk {
  anchor: string | null;
  eof: boolean;
  oLines: string[];
  nLines: NewLine[];
}

/** A chunk's body lines as old and new lines: context lines are both, - lines are old, + lines are new. */
function chunkLines(body: BodyLine[]): { oLines: string[]; nLines: NewLine[] } {
  const oLines: string[] = [];
  const nLines: NewLine[] = [];
  for (const b of body) {
    if (b.kind === ' ' || b.kind === '') {
      // An empty line is a context line whose text is empty.
      oLines.push(b.text);
      nLines.push({ text: b.text, ctx: oLines.length - 1 });
    } else if (b.kind === '-') {
      oLines.push(b.text);
    } else if (b.kind === '+') {
      nLines.push({ text: b.text, ctx: null });
    }
  }
  return { oLines, nLines };
}

const isSectionStart = (l: string): boolean =>
  l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ');

/** A V4A patch's sections, each one edit. The first non-blank line is *** Begin Patch and the last non-blank
 *  line is *** End Patch; blank lines between sections are ignored, and blank lines at the end of a section
 *  are not part of it. */
function v4aEdits(lines: string[], strict: boolean, moves: Move[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = -1;
  for (let k = lines.length - 1; k >= 0; k--) {
    if (!blank(lines[k]!)) {
      last = k;
      break;
    }
  }
  if (lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);
  const edits: Edit[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i]!;
    if (blank(line)) {
      i++;
      continue;
    }
    if (line.startsWith('*** Add File: ')) {
      const path = line.slice(14).trim();
      let j = i + 1;
      const raw: string[] = [];
      while (j < last && !isSectionStart(lines[j]!)) {
        raw.push(lines[j]!);
        j++;
      }
      while (raw.length > 0 && blank(raw[raw.length - 1]!)) raw.pop();
      const fileLines: string[] = [];
      for (let k = 0; k < raw.length; k++) {
        const l = raw[k]!;
        if (l === '') fileLines.push('');
        else if (l.startsWith('+')) fileLines.push(l.slice(1));
        else throw new Exit2(`error: parse ${i + 2 + k}`);
      }
      edits.push((tree) => {
        if (!safe(path)) throw new Fail('unsafe', path);
        if (tree.get(path) !== null) throw new Fail('exists', path);
        tree.now.set(path, fileLines.length === 0 ? '' : fileLines.join('\n') + '\n');
      });
      i = j;
      continue;
    }
    if (line.startsWith('*** Delete File: ')) {
      const path = line.slice(17).trim();
      let j = i + 1;
      while (j < last && !isSectionStart(lines[j]!)) {
        if (!blank(lines[j]!)) throw new Exit2(`error: parse ${j + 1}`);
        j++;
      }
      edits.push((tree) => {
        if (!safe(path)) throw new Fail('unsafe', path);
        if (tree.get(path) === null) throw new Fail('missing', path);
        tree.now.set(path, null);
      });
      i = j;
      continue;
    }
    if (line.startsWith('*** Update File: ')) {
      const path = line.slice(17).trim();
      const updateLine = i;
      let j = i + 1;
      let moveTo: string | null = null;
      if (j < last && lines[j]!.startsWith('*** Move to: ')) {
        moveTo = lines[j]!.slice(13).trim();
        j++;
      }
      const rawStart = j;
      const raw: string[] = [];
      while (j < last && !isSectionStart(lines[j]!)) {
        raw.push(lines[j]!);
        j++;
      }
      while (raw.length > 0 && blank(raw[raw.length - 1]!)) raw.pop();
      const chunks: VChunk[] = [];
      let cur: { anchor: string | null; eof: boolean; body: BodyLine[] } | null = null;
      let started = false;
      const closeChunk = (): void => {
        if (cur === null) return;
        const { oLines, nLines } = chunkLines(cur.body);
        chunks.push({ anchor: cur.anchor, eof: cur.eof, oLines, nLines });
        cur = null;
      };
      for (let k = 0; k < raw.length; k++) {
        const l = raw[k]!;
        const lineNo = rawStart + k + 1;
        if (l.startsWith('@@')) {
          closeChunk();
          const anchor = l.slice(2).trim();
          cur = { anchor: anchor === '' ? null : anchor, eof: false, body: [] };
          started = true;
          continue;
        }
        if (l.trim() === '*** End of File') {
          if (cur === null) throw new Exit2(`error: parse ${lineNo}`);
          cur.eof = true;
          closeChunk();
          continue;
        }
        if (l === '' || l[0] === ' ' || l[0] === '-' || l[0] === '+') {
          if (cur === null) {
            if (started) {
              if (l === '') continue;
              throw new Exit2(`error: parse ${lineNo}`);
            }
            cur = { anchor: null, eof: false, body: [] };
            started = true;
          }
          cur.body.push({ kind: (l === '' ? '' : l[0]) as BodyLine['kind'], text: l === '' ? '' : l.slice(1) });
          continue;
        }
        throw new Exit2(`error: parse ${lineNo}`);
      }
      closeChunk();
      if (chunks.length === 0 && moveTo === null) throw new Exit2(`error: parse ${updateLine + 1}`);
      edits.push((tree) => {
        if (!safe(path)) throw new Fail('unsafe', path);
        if (moveTo !== null && !safe(moveTo)) throw new Fail('unsafe', moveTo);
        const before = tree.get(path);
        if (before === null) throw new Fail('missing', path);
        if (moveTo !== null && tree.get(moveTo) !== null) throw new Fail('exists', moveTo);
        const state = parseFile(before);
        let cursor = 0;
        for (const ch of chunks) {
          if (ch.anchor !== null) {
            const a = matchAt(state.lines, [ch.anchor], strict, cursor);
            if (a === null) throw new Fail('no-match', path);
            cursor = a.positions[0]! + 1;
          }
          if (ch.oLines.length > 0) {
            const m = matchAt(state.lines, ch.oLines, strict, cursor, ch.eof);
            if (m === null) throw new Fail('no-match', path);
            const pos = m.positions[0]!;
            const rendered = renderNew(state.lines, pos, ch.oLines, ch.nLines, m.tier);
            state.lines.splice(pos, ch.oLines.length, ...rendered);
            cursor = pos + rendered.length;
          } else {
            const at = ch.eof ? state.lines.length : cursor;
            state.lines.splice(at, 0, ...ch.nLines.map((n) => n.text));
            cursor = at + ch.nLines.length;
          }
        }
        const out = serializeFile(state);
        if (moveTo !== null) {
          tree.now.set(moveTo, out);
          tree.now.set(path, null);
          moves.push({ old: path, next: moveTo });
        } else {
          tree.now.set(path, out);
        }
      });
      i = j;
      continue;
    }
    throw new Exit2(`error: parse ${i + 1}`);
  }
  if (edits.length === 0) throw new Exit2('error: parse 1');
  return edits;
}

/** Run the CLI on its arguments: the exit code and the lines for standard output. Usage errors beat everything
 *  else, and a parse error is reported before any file is looked at. A failed edit changes nothing, and the
 *  edits after it still run, so every failure is reported. When any edit failed, no file is written. */
function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  const moves: Move[] = [];
  let edits: Edit[];
  try {
    edits =
      kind === 'json'
        ? jsonEdits(text, lines)
        : kind === 'unified'
          ? unifiedEdits(lines, strict, moves)
          : kind === 'search-replace'
            ? srEdits(lines, strict, moves)
            : v4aEdits(lines, strict, moves);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }
  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      fails.push(e);
    }
  }
  if (fails.length > 0) return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  const keys = [...tree.now.keys()].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
  const rows: { key: string; text: string }[] = [];
  for (const p of keys) {
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    rows.push({ key: p, text: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  for (const m of moves) {
    if (tree.original(m.old) !== null && tree.get(m.old) === null && tree.original(m.next) === null && tree.get(m.next) !== null) {
      const di = rows.findIndex((r) => r.text === `D ${m.old}`);
      const ai = rows.findIndex((r) => r.text === `A ${m.next}`);
      if (di >= 0 && ai >= 0) {
        rows.splice(Math.max(di, ai), 1);
        rows.splice(Math.min(di, ai), 1);
        rows.push({ key: m.next, text: `R ${m.old} -> ${m.next}` });
      }
    }
  }
  rows.sort((a, b) => Buffer.compare(Buffer.from(a.key), Buffer.from(b.key)));
  if (!check) {
    for (const p of keys) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  return { code: 0, out: [...rows.map((r) => r.text), `ok: ${rows.length} ${rows.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
