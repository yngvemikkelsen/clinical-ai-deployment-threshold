// Build one .docx per multimedia appendix.
//
// Design decisions, so a later reader knows why:
//  - Landscape US Letter for every appendix that carries a wide table; JMIR
//    renders appendices as supplied, and a 14-column table in portrait is
//    unreadable.
//  - Appendix 1 has 24 columns, which does not fit any page. It is therefore
//    split into three logically separate tables over the same 55 rows
//    (identification, evidence, classification) rather than shrunk to
//    illegibility.
//  - Every table sets columnWidths on the table AND width on each cell, both
//    in DXA. Percentage widths break in Google Docs.
//  - ShadingType.CLEAR for header fill; SOLID renders black.
//  - Header rows repeat across pages and rows are not split mid-row, so a
//    long appendix stays readable.

const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageOrientation
} = d;

const APP = 'appendices';
const OUT = '/mnt/user-data/outputs/multimedia_appendices';
fs.mkdirSync(OUT, { recursive: true });

// US Letter in DXA (1440 = 1 inch)
const LETTER = { width: 12240, height: 15840 };
const MARGIN = 1080;                                  // 0.75"
const USABLE_LANDSCAPE = LETTER.height - 2 * MARGIN;  // 13,680
const USABLE_PORTRAIT  = LETTER.width  - 2 * MARGIN;  // 10,080

const GREY = 'F2F2F2';
const RULE = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const BORDERS = { top: RULE, bottom: RULE, left: RULE, right: RULE,
                  insideHorizontal: RULE, insideVertical: RULE };

function parseCsv(file) {
  const text = fs.readFileSync(path.join(APP, file), 'utf8');
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') q = false;
      else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else if (c !== '\r') cell += c;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.some(x => x !== ''));
}

function widthsFor(header, body, usable) {
  // proportional to the longest content in each column, floored so narrow
  // columns stay legible and capped so one long free-text column cannot
  // squeeze the rest to nothing
  const len = header.map((h, i) => {
    let m = String(h).length;
    for (const r of body) m = Math.max(m, String(r[i] ?? '').length);
    return Math.min(Math.max(m, 6), 46);
  });
  const tot = len.reduce((a, b) => a + b, 0);
  const w = len.map(l => Math.round(usable * l / tot));
  w[w.length - 1] += usable - w.reduce((a, b) => a + b, 0);   // exact sum
  return w;
}

function tableFrom(rows, usable, fontSize) {
  const header = rows[0], body = rows.slice(1);
  const cw = widthsFor(header, body, usable);
  const mk = (cells, isHeader) => new TableRow({
    cantSplit: true,
    tableHeader: isHeader,
    children: cells.map((c, i) => new TableCell({
      width: { size: cw[i], type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, fill: GREY, color: 'auto' } : undefined,
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      children: [new Paragraph({
        spacing: { before: 0, after: 0 },
        children: [new TextRun({ text: String(c ?? ''), bold: !!isHeader,
                                 size: fontSize, font: 'Calibri' })]
      })]
    }))
  });
  return new Table({
    columnWidths: cw,
    width: { size: usable, type: WidthType.DXA },
    borders: BORDERS,
    rows: [mk(header, true), ...body.map(r => mk(r, false))]
  });
}

// Structured legend format, so rendering is deterministic:
//   "## x"  subheading      "| a | b |"  table row      "- x"  list item
//   "> x"   block quote     blank line   paragraph break
// Paragraph text is NOT hard-wrapped in the source, so nothing is reflowed
// and no wrapped line can be misread as a heading or a list item.
function legendBlocks(file, usable, tableScale) {
  if (tableScale === undefined) tableScale = 0.72;
  const fp = path.join('mma_src', file);
  if (!fs.existsSync(fp)) return [];
  const lines = fs.readFileSync(fp, 'utf8').split('\n');
  const out = [];
  let tbl = [];
  const flushTable = () => {
    if (!tbl.length) return;
    out.push(tableFrom(tbl, Math.round(usable * tableScale), 17));
    tbl = [];
  };
  for (const raw of lines) {
    const l = raw.trim();
    if (l.startsWith('|')) {
      tbl.push(l.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
      continue;
    }
    flushTable();
    if (l === '') continue;
    if (l.startsWith('## ')) {
      out.push(new Paragraph({
        spacing: { before: 240, after: 90 },
        children: [new TextRun({ text: l.slice(3), bold: true, size: 21, font: 'Calibri' })]
      }));
    } else if (l.startsWith('- ')) {
      out.push(new Paragraph({
        spacing: { after: 70, line: 264 },
        indent: { left: 360, hanging: 200 },
        children: [new TextRun({ text: '\u2022  ' + l.slice(2), size: 20, font: 'Calibri' })]
      }));
    } else if (l.startsWith('> ')) {
      out.push(new Paragraph({
        spacing: { before: 80, after: 140, line: 252 },
        indent: { left: 420 },
        children: [new TextRun({ text: l.slice(2), size: 18, font: 'Consolas' })]
      }));
    } else {
      out.push(new Paragraph({
        spacing: { after: 130, line: 276 },
        children: [new TextRun({ text: l, size: 20, font: 'Calibri' })]
      }));
    }
  }
  flushTable();
  return out;
}

function title(text) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [new TextRun({ text, bold: true, size: 26, font: 'Calibri' })]
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text, size: 20, font: 'Calibri', italics: true })]
  });
}
function tableTitle(text) {
  return new Paragraph({
    spacing: { before: 240, after: 100 },
    children: [new TextRun({ text, bold: true, size: 20, font: 'Calibri' })]
  });
}

function build(name, landscape, blocks) {
  const usable = landscape ? USABLE_LANDSCAPE : USABLE_PORTRAIT;
  const doc = new Document({
    styles: { default: { document: { run: { font: 'Calibri', size: 21 } } } },
    sections: [{
      properties: {
        page: {
          size: landscape
            ? { width: LETTER.width, height: LETTER.height, orientation: PageOrientation.LANDSCAPE }
            : LETTER,
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN }
        }
      },
      children: blocks(usable)
    }]
  });
  return Packer.toBuffer(doc).then(b => {
    fs.writeFileSync(path.join(OUT, name), b);
    console.log(`  ${name}  ${(b.length / 1024).toFixed(0)} KB`);
  });
}

// ------------------------------------------------------------------ MMA 1
// 24 columns cannot fit a page. Split over the same 55 rows into three
// tables that each answer a different question.
const A1 = parseCsv('appendix1_system_extraction.csv');
const a1h = A1[0], a1b = A1.slice(1);
const pick = names => {
  const idx = names.map(n => a1h.indexOf(n)).filter(i => i >= 0);
  return [idx.map(i => a1h[i]), ...a1b.map(r => idx.map(i => r[i]))];
};

const mma1 = build('Multimedia_Appendix_1.docx', true, u => [
  title('Multimedia Appendix 1. System extraction frame and coding definitions'),
  caption('All 55 clinical retrieval-augmented generation systems, deduplicated from two systematic reviews. Tables 1.1 to 1.4 present the same 55 systems and summary counts from four complementary angles: what each reports, what could be inferred and on what evidence, how each is classified under the four prespecified classification assumptions, and the distribution of interface classes across the frame.'),
  ...legendBlocks('a1_definitions.txt', u),
  tableTitle('Table 1.1. Reported implementation characteristics'),
  tableFrom(pick(['Canonical ID', 'System', 'Source frame',
                  'Reported operational encoder/retriever', 'Interface class',
                  'Reported pooling']), u, 15),
  tableTitle('Table 1.2. Inferred extraction and the evidence for it'),
  tableFrom(pick(['Canonical ID', 'Inferred extraction/representation',
                  'Basis for inference', 'Primary-source evidence',
                  'Extraction step established', 'Coding stability',
                  'Superseded coding rationale']), u, 14),
  tableTitle('Table 1.3. Classification under each prespecified assumption'),
  tableFrom(pick(['Canonical ID', 'Confirmed affected', 'Confidence',
                  'Rung 1: no assumptions',
                  'Rung 2: contrastive training implies non-membership',
                  'Rung 3: vendor claims accepted',
                  'Rung 4: non-dense excluded']), u, 15),
  tableTitle('Table 1.4. Interface class counts'),
  tableFrom(parseCsv('appendix1_summary.csv'), Math.round(u * 0.5), 18),
]);

// ------------------------------------------------------------------ MMA 2
const mma2 = build('Multimedia_Appendix_2.docx', true, u => [
  title('Multimedia Appendix 2. Corpus construction, section extraction, and the complete transport matrix'),
  caption('Baseline, corrected, and change in mean reciprocal rank at cutoff 10 for every configuration, corpus, query format, and document variant, with nominal and benchmark-response subgroup assignment.'),
  ...legendBlocks('a2_legend.txt', u),
  tableTitle('Table 2.1. Cell-level transport matrix'),
  tableFrom(parseCsv('appendix2_transport_matrix.csv'), u, 15),
]);

// ------------------------------------------------------------------ MMA 3
const mma3 = build('Multimedia_Appendix_3.docx', true, u => [
  title('Multimedia Appendix 3. Local-screen resampling results'),
  caption('Sign-agreement rates by configuration, corpus, query format, and document sample size. The reference sign is the configuration\u2019s full-sample measured effect on that corpus. Intervals are Wilson intervals on the resampling proportion and quantify Monte Carlo precision for these corpora, not spread across sites: the 500 draws are overlapping subsamples of two fixed 100-document corpora, not independent deployment environments. Sample size refers to a joint fit-and-evaluation sample, so the transform-fitting and effect-estimation requirements are not separately identified.'),
  tableTitle('Table 3.1. Resampling agreement by sample size'),
  tableFrom(parseCsv('appendix3_screen_resampling.csv'), u, 14),
]);

// ------------------------------------------------------------------ MMA 4
const mma4 = build('Multimedia_Appendix_4.docx', true, u => [
  title('Multimedia Appendix 4. Query-generation robustness and the full generation protocol'),
  caption('Paired comparisons under the published and locally generated query sets on the derivation study\u2019s own MTSamples sample, holding documents, configurations and all other settings constant, followed by the generation protocol verbatim.'),
  tableTitle('Table 4.1. Paired comparison of query sets'),
  tableFrom(parseCsv('appendix4_query_generator.csv'), u, 15),
  ...legendBlocks('a4_protocol.txt', u),
]);

// ------------------------------------------------------------------ MMA 5
const mma5 = build('Multimedia_Appendix_5.docx', true, u => [
  title('Multimedia Appendix 5. Decision-model parameters, regularisation sweep, and per-condition effects'),
  caption('Every parameter with its value, distribution or range, source, and whether it is empirical, scenario-based, or illustrative; the regularisation sweep with response-defined grouping at each value; and the per-condition effect estimates from which the group means are computed.'),
  tableTitle('Table 5.1. Model parameters'),
  tableFrom(parseCsv('appendix5_parameters.csv'), u, 15),
  tableTitle('Table 5.2. Regularisation sweep under response-defined grouping'),
  caption('Group membership is determined at each value by the sign of the measured mean effect, so group sizes change across the sweep. This is the structural sensitivity reported in place of a sampling interval on the threshold.'),
  tableFrom(parseCsv('appendix5_epsilon_sweep.csv'), Math.round(u * 0.8), 17),
  tableTitle('Table 5.3. Per-condition effect estimates'),
  tableFrom(parseCsv('appendix5_epsilon_percondition.csv'), u, 14),
]);

// ------------------------------------------------------------------ MMA 6
const mma6 = build('Multimedia_Appendix_6.docx', true, u => [
  title('Multimedia Appendix 6. Code-search boundary replication'),
  caption('Replication of the published semantic code-search experiment from the authors\u2019 released code and data, undertaken to test whether the sign-changing antecedent holds in a domain sharing no models, corpora or evaluation conventions with clinical retrieval. Baseline retrieval reproduced the published values to within 0.001 across all 18 primary cells; the reported subgroup sign structure did not reproduce, and the generality claim is withdrawn rather than supported.'),
  tableTitle('Table 6.1. Baseline replication against published values'),
  tableFrom(parseCsv('appendix6_baseline_replication.csv'), Math.round(u * 0.85), 17),
  tableTitle('Table 6.2. Whitening effects by configuration and language'),
  tableFrom(parseCsv('appendix6_codesearch_full.csv'), u, 15),
  tableTitle('Table 6.3. Regularisation grid'),
  tableFrom(parseCsv('appendix6_epsilon_grid.csv'), Math.round(u * 0.85), 17),
]);

// ------------------------------------------------------------------ MMA 7
const mma7 = build('Multimedia_Appendix_7.docx', true, u => [
  title('Multimedia Appendix 7. Completed CHEERS 2022 checklist'),
  caption('Consolidated Health Economic Evaluation Reporting Standards 2022, all 28 items, with reported status and the manuscript location of each.'),
  ...legendBlocks('a7_cheers2022.txt', u, 1.0),
]);

// ------------------------------------------------------------------ MMA 8
const mma8 = build('Multimedia_Appendix_8.docx', true, u => [
  title('Multimedia Appendix 8. Completed CHEERS-AI checklist'),
  caption('Consolidated Health Economic Evaluation Reporting Standards for Interventions That Use Artificial Intelligence: the 28 CHEERS 2022 items with 8 AI-specific elaborations, plus the 10 AI extension items, with reported status and manuscript location of each.'),
  ...legendBlocks('a8_cheersai.txt', u, 1.0),
]);

Promise.all([mma1, mma2, mma3, mma4, mma5, mma6, mma7, mma8])
  .then(() => console.log('done'))
  .catch(e => { console.error(e); process.exit(1); });
