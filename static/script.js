async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function renderTableList(container, names) {
  container.innerHTML = '';
  if (!names.length) {
    container.textContent = 'No tables found';
    return;
  }
  names.forEach((name) => {
    const btn = document.createElement('button');
    btn.textContent = name;
    btn.onclick = () => loadPreview(name);
    container.appendChild(btn);
  });
}

function renderPreview(container, payload) {
  container.innerHTML = '';
  const { columns, rows } = payload;
  if (!rows.length) {
    container.textContent = 'No rows';
    return;
  }
  const table = document.createElement('table');
  table.className = 'table';
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  columns.forEach((c) => {
    const th = document.createElement('th');
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    columns.forEach((c) => {
      const td = document.createElement('td');
      td.textContent = r[c];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

async function loadTables() {
  try {
    const data = await fetchJSON('/api/tables');
    const names = [...data.tables, ...data.views];
    renderTableList(document.getElementById('tables'), names);
  } catch (e) {
    document.getElementById('tables').textContent = String(e);
  }
}

async function loadPreview(name) {
  const container = document.getElementById('preview');
  container.textContent = 'Loading...';
  try {
    const data = await fetchJSON(`/api/table/${encodeURIComponent(name)}?limit=50`);
    renderPreview(container, data);
  } catch (e) {
    container.textContent = String(e);
  }
}

function drawDiagramFromModel(model) {
  // Expecting minimal structure: { entities: [{ name, fields: [name, ...] }], relations: [{ from, to, on }] }
  const box = (title, fields) => {
    const maxLen = Math.max(title.length, ...fields.map((f) => f.length), 4);
    const pad = (s) => s.padEnd(maxLen, ' ');
    const top = '+' + '-'.repeat(maxLen + 2) + '+\n';
    const header = `| ${pad(title)} |\n`;
    const sep = '+' + '-'.repeat(maxLen + 2) + '+\n';
    const rows = fields.map((f) => `| ${pad(f)} |\n`).join('');
    return top + header + sep + rows + sep;
  };

  const lines = [];
  if (Array.isArray(model.entities)) {
    model.entities.forEach((e) => {
      const fields = Array.isArray(e.fields) ? e.fields : [];
      lines.push(box(String(e.name || 'Entity'), fields.map(String)));
    });
  }
  if (Array.isArray(model.relations) && model.relations.length) {
    lines.push('\nRelations:');
    model.relations.forEach((r) => {
      lines.push(`- ${r.from} -> ${r.to} on ${r.on || '?'} `);
    });
  }
  document.getElementById('diagram').textContent = lines.join('\n');
}

function setupModelUpload() {
  const input = document.getElementById('modelFile');
  input.addEventListener('change', async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      drawDiagramFromModel(json);
    } catch (e) {
      document.getElementById('diagram').textContent = `Invalid JSON: ${e}`;
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  loadTables();
  setupModelUpload();
});
