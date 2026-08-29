/* Tiny, dependency-free CommonMark-ish renderer for answer_md.
 * Supports: fenced code ```lang, 3+ backticks; inline code; headings 1-4;
 * bold/italic/links; images; blockquote; ordered/unordered list; tables (GFM).
 * Intentionally conservative — the data is trusted, but we escape HTML first
 * so raw HTML in crawled content never runs script in the browser.
 */
(function (global) {
    'use strict';

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatInline(text, allowImage) {
        // images first: ![alt](url "title?")
        let t = text;
        if (allowImage) {
            t = t.replace(/!\[([^\]]*)\]\((<[^>]+>|[^)\s]+)(?:\s+"([^"]*)")?\)/g, (_m, alt, url, title) => {
                let u = url;
                if (u.charAt(0) === '<' && u.charAt(u.length - 1) === '>') u = u.slice(1, -1);
                return `<img alt="${escapeHtml(alt)}" src="${escapeHtml(u)}"${title ? ' title="' + escapeHtml(title) + '"' : ''} referrerpolicy="no-referrer" loading="lazy" />`;
            });
        }
        // links: [text](url "title?")
        t = t.replace(/\[([^\]]+)\]\((<[^>]+>|[^)\s]+)(?:\s+"([^"]*)")?\)/g, (_m, txt, url, title) => {
            let u = url;
            if (u.charAt(0) === '<' && u.charAt(u.length - 1) === '>') u = u.slice(1, -1);
            return `<a href="${escapeHtml(u)}"${title ? ' title="' + escapeHtml(title) + '"' : ''} target="_blank" rel="noopener noreferrer">${txt}</a>`;
        });
        // inline code `x` or ``x with ` inside``
        let out = '';
        let i = 0;
        while (i < t.length) {
            let m = /`+/.exec(t.slice(i));
            if (!m) { out += t.slice(i); break; }
            const tick = m[0];
            const start = i + m.index;
            const closeIdx = t.indexOf(tick, start + tick.length);
            if (closeIdx === -1) { out += t.slice(i); break; }
            out += t.slice(i, start) + '<code>' + escapeHtml(t.slice(start + tick.length, closeIdx)) + '</code>';
            i = closeIdx + tick.length;
        }
        t = out;
        // bold **x** / __x__
        t = t.replace(/(\*\*|__)([^\n]*?)\1/g, '<strong>$2</strong>');
        // italic *x* / _x_ (require non-underscore boundary for _italic_)
        t = t.replace(/(^|[^*])\*([^\n*]+)\*(?!\*)/g, '$1<em>$2</em>');
        // simple st ~~x~~
        t = t.replace(/~~([^\n]*?)~~/g, '<s>$1</s>');
        return t;
    }

    function renderTable(rows) {
        if (rows.length < 2) return '<p>' + rows.map(r => escapeHtml(r)).join('<br>') + '</p>';
        const split = (line) => {
            const trimmed = line.replace(/^\s*\|/, '').replace(/\|\s*$/, '');
            if (!trimmed) return [];
            return trimmed.split('|').map(c => c.trim());
        };
        const header = split(rows[0]);
        const sep = split(rows[1]);
        // must look like GFM sep row
        const isSep = sep.length === header.length && sep.every(c => /^:?-{3,}:?$/.test(c));
        if (!isSep) return '<p>' + escapeHtml(rows.join('\n')) + '</p>';
        const aligns = sep.map(c => {
            const left = c.startsWith(':');
            const right = c.endsWith(':');
            return left && right ? 'center' : right ? 'right' : left ? 'left' : null;
        });
        const bodyRows = rows.slice(2).map(split);
        const cell = (txt, al) => `<td${al ? ` style="text-align:${al}"` : ''}>${formatInline(escapeHtml(txt))}</td>`;
        const hdr = header.map((h, i) => `<th${aligns[i] ? ` style="text-align:${aligns[i]}"` : ''}>${formatInline(escapeHtml(h))}</th>`).join('');
        const body = bodyRows.map(r => '<tr>' + r.map((c, i) => cell(c, aligns[i])).join('') + '</tr>').join('');
        return '<table><thead><tr>' + hdr + '</tr></thead><tbody>' + body + '</tbody></table>';
    }

    function render(md) {
        if (md == null) return '';
        const text = String(md).replace(/\r\n?/g, '\n');
        const lines = text.split('\n');
        let html = '';
        let i = 0;
        const flushParagraph = (buf) => {
            if (!buf.length) return;
            html += '<p>' + formatInline(escapeHtml(buf.join('\n')), true) + '</p>';
        };

        while (i < lines.length) {
            const line = lines[i];

            // fenced code ```lang / ~~~
            const fence = /^(\s*)(```+|~~~+)(\s*([\w+-]*))?\s*$/.exec(line);
            if (fence) {
                flushParagraph(p); p = [];
                const marker = fence[2][0];
                const needed = fence[2].length;
                const lang = fence[4] || '';
                const buf = [];
                i++;
                while (i < lines.length) {
                    const ln = lines[i];
                    const close = new RegExp('^(\\s*)(' + marker + '){' + needed + ',}\\s*$');
                    if (close.test(ln)) break;
                    buf.push(ln);
                    i++;
                }
                html += `<pre><code${lang ? ` class="language-${escapeHtml(lang)}"` : ''}>${escapeHtml(buf.join('\n'))}</code></pre>`;
                i++; // consume closing fence line
                continue;
            }

            // blank line
            if (/^\s*$/.test(line)) {
                flushParagraph(p); p = [];
                i++;
                continue;
            }

            // heading
            const heading = /^(#{1,4})\s+(.*)$/.exec(line);
            if (heading) {
                flushParagraph(p); p = [];
                const lvl = heading[1].length;
                html += `<h${lvl}>${formatInline(escapeHtml(heading[2].trim()), true)}</h${lvl}>`;
                i++;
                continue;
            }

            // blockquote (collect contiguous lines starting with > )
            if (/^\s*>\s?/.test(line)) {
                flushParagraph(p); p = [];
                const quoteLines = [];
                while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
                    quoteLines.push(lines[i].replace(/^\s*>\s?/, ''));
                    i++;
                }
                html += '<blockquote>' + render(quoteLines.join('\n')) + '</blockquote>';
                continue;
            }

            // table detection: header row has pipes; next line is separator
            if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
                flushParagraph(p); p = [];
                const rows = [line, lines[i + 1]];
                i += 2;
                while (i < lines.length && /\|/.test(lines[i]) && /\S/.test(lines[i])) {
                    rows.push(lines[i]);
                    i++;
                }
                html += renderTable(rows);
                continue;
            }

            // list: unordered (- * +) or ordered (1. 2)
            const listStarter = /^(\s*)([-*+]|\d+\.)\s+(.*)$/.exec(line);
            if (listStarter) {
                flushParagraph(p); p = [];
                const ordered = /^\d+\.$/.test(listStarter[2]);
                const tag = ordered ? 'ol' : 'ul';
                html += `<${tag}>`;
                while (i < lines.length) {
                    const m = /^(\s*)([-*+]|\d+\.)\s+(.*)$/.exec(lines[i]);
                    if (!m) break;
                    html += '<li>' + formatInline(escapeHtml(m[3]), true) + '</li>';
                    i++;
                }
                html += `</${tag}>`;
                continue;
            }

            // thematic break --- / ___ / ***
            if (/^\s*([-*_])\s*\1\s*\1[\s\S]*$/.test(line) && /^[\s*-_*]+$/.test(line)) {
                flushParagraph(p); p = [];
                html += '<hr>';
                i++;
                continue;
            }

            // accumulate paragraph
            p.push(line);
            i++;
        }
        flushParagraph(p);
        return html;
    }

    global.AppleMarkdown = { render };
})(window);
