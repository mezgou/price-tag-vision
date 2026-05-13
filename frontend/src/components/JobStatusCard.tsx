import type { Job } from '../types/job'

type JobStatusCardProps = {
  csvDownloadUrl: string | null
  isPolling: boolean
  job: Job
  refreshError: string | null
}

function formatDateTime(value: string): string {
  return new Date(value).toLocaleString()
}

function renderValue(value: unknown): string {
  if (value === null || value === undefined) {
    return 'n/a'
  }

  if (typeof value === 'object') {
    return JSON.stringify(value)
  }

  return String(value)
}

export function JobStatusCard({
  csvDownloadUrl,
  isPolling,
  job,
  refreshError,
}: JobStatusCardProps) {
  const progress = Math.min(100, Math.max(0, job.progress))
  const statusClassName = `status-chip status-${job.status}`
  const statsEntries = job.stats_json ? Object.entries(job.stats_json) : []

  return (
    <section className="panel job-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Current Job</p>
          <h2>{job.original_filename}</h2>
        </div>
        <div className={statusClassName}>{job.status}</div>
      </div>

      <div className="job-meta-row">
        <span>Job ID: {job.id}</span>
        <span>{isPolling ? 'Polling backend for updates' : 'Polling stopped'}</span>
      </div>

      <div className="progress-block">
        <div className="progress-meta">
          <strong>Progress</strong>
          <span>{progress}%</span>
        </div>
        <div className="progress-track" aria-hidden="true">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="detail-grid">
        <div className="detail-item">
          <span className="detail-label">Stage</span>
          <strong>{job.stage ?? 'n/a'}</strong>
        </div>
        <div className="detail-item">
          <span className="detail-label">Message</span>
          <strong>{job.message ?? 'No message yet.'}</strong>
        </div>
        <div className="detail-item">
          <span className="detail-label">Pipeline</span>
          <strong>
            {job.pipeline_name} / {job.pipeline_version}
          </strong>
        </div>
        <div className="detail-item">
          <span className="detail-label">Updated</span>
          <strong>{formatDateTime(job.updated_at)}</strong>
        </div>
      </div>

      {job.error ? <div className="banner banner-error">{job.error}</div> : null}
      {refreshError ? <div className="banner banner-warning">{refreshError}</div> : null}

      {statsEntries.length > 0 ? (
        <div className="stats-block">
          <p className="eyebrow">Worker Stats</p>
          <div className="stats-grid">
            {statsEntries.map(([key, value]) => (
              <div key={key} className="stats-item">
                <span className="detail-label">{key}</span>
                <strong>{renderValue(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="job-footer">
        <span>Created {formatDateTime(job.created_at)}</span>
        {csvDownloadUrl ? (
          <a
            className="button-link"
            href={csvDownloadUrl}
            rel="noreferrer"
            target="_blank"
          >
            Download CSV
          </a>
        ) : (
          <span>CSV will appear after a successful run</span>
        )}
      </div>
    </section>
  )
}
