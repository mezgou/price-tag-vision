import type { PreviewPayload } from '../types/job'

type PreviewTableProps = {
  preview: PreviewPayload
}

type PreviewRow = Record<string, unknown>

function isRecord(value: unknown): value is PreviewRow {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function extractRows(preview: PreviewPayload): PreviewRow[] | null {
  if (Array.isArray(preview) && preview.every(isRecord)) {
    return preview
  }

  if (!isRecord(preview)) {
    return null
  }

  const candidates = [preview.rows, preview.list]
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.every(isRecord)) {
      return candidate
    }
  }

  return null
}

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }

  if (typeof value === 'object') {
    return JSON.stringify(value)
  }

  return String(value)
}

export function PreviewTable({ preview }: PreviewTableProps) {
  const rows = extractRows(preview)

  if (rows) {
    const columns = Array.from(
      new Set(rows.flatMap((row) => Object.keys(row))),
    )

    if (rows.length === 0 || columns.length === 0) {
      return <p className="empty-copy">Preview payload contains no rows yet.</p>
    }

    return (
      <div className="preview-table-shell">
        <table className="preview-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>
                    {stringifyValue(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return <pre className="json-block">{JSON.stringify(preview, null, 2)}</pre>
}
