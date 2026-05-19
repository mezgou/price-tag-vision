import { useEffect, useState } from 'react'
import './App.css'
import { API_BASE_URL, apiClient, getApiErrorMessage } from './api/client'
import {
  createJob,
  getCsvDownloadUrl,
  getJob,
  getJobPreview,
} from './api/jobs'
import { JobStatusCard } from './components/JobStatusCard'
import { PreviewTable } from './components/PreviewTable'
import { UploadDropzone } from './components/UploadDropzone'
import type { Job, PreviewPayload } from './types/job'

type ServiceStatus = {
  checked_at: string
  detail: string
  status: string
}

type StatusPayload = {
  checked_at: string
  service: string
  services: Record<string, ServiceStatus>
  status: string
}

const POLL_INTERVAL_MS = 1_500

const TERMINAL_JOB_STATUSES = new Set(['succeeded', 'failed', 'cancelled'])

const serviceMeta = [
  { key: 'frontend', label: 'Frontend', href: window.location.origin },
  { key: 'backend', label: 'Backend', href: API_BASE_URL },
  { key: 'worker', label: 'Workers', href: null },
  { key: 'vision_service', label: 'Vision Service', href: 'http://localhost:9001' },
  { key: 'postgres', label: 'PostgreSQL', href: null },
  { key: 'redis', label: 'Redis', href: null },
  { key: 'minio', label: 'MinIO', href: 'http://localhost:9002' },
] as const

function App() {
  const [job, setJob] = useState<Job | null>(null)
  const [preview, setPreview] = useState<PreviewPayload | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isArtifactsLoading, setIsArtifactsLoading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [resumeError, setResumeError] = useState<string | null>(null)
  const [jobRefreshError, setJobRefreshError] = useState<string | null>(null)
  const [artifactError, setArtifactError] = useState<string | null>(null)
  const [systemStatus, setSystemStatus] = useState<StatusPayload | null>(null)
  const [systemError, setSystemError] = useState<string | null>(null)

  const initialJobId = new URLSearchParams(window.location.search).get('jobId')
  const jobId = job?.id ?? null
  const jobStatus = job?.status ?? null
  const isPolling = job ? !TERMINAL_JOB_STATUSES.has(job.status) : false
  const csvDownloadUrl =
    job && job.status === 'succeeded' ? getCsvDownloadUrl(job.id) : null

  useEffect(() => {
    if (!initialJobId) {
      return
    }

    let isCancelled = false

    const restoreJob = async () => {
      try {
        const existingJob = await getJob(initialJobId)
        if (isCancelled) {
          return
        }

        setJob(existingJob)
        setResumeError(null)
      } catch (error) {
        if (isCancelled) {
          return
        }

        setResumeError(
          getApiErrorMessage(error, 'Failed to restore job from the URL.'),
        )
      }
    }

    void restoreJob()

    return () => {
      isCancelled = true
    }
  }, [initialJobId])

  useEffect(() => {
    const url = new URL(window.location.href)
    if (jobId) {
      url.searchParams.set('jobId', jobId)
    } else {
      url.searchParams.delete('jobId')
    }

    const nextUrl = `${url.pathname}${url.search}${url.hash}`
    window.history.replaceState({}, '', nextUrl)
  }, [jobId])

  useEffect(() => {
    let isCancelled = false

    const loadSystemStatus = async () => {
      try {
        const response = await apiClient.get<StatusPayload>('/api/system/status', {
          validateStatus: () => true,
        })

        if (isCancelled) {
          return
        }

        setSystemStatus(response.data)
        setSystemError(
          response.status >= 400
            ? 'One or more services are still unhealthy.'
            : null,
        )
      } catch (error) {
        if (isCancelled) {
          return
        }

        setSystemError(
          getApiErrorMessage(error, 'System status request failed.'),
        )
      }
    }

    void loadSystemStatus()
    const intervalId = window.setInterval(() => {
      void loadSystemStatus()
    }, 10_000)

    return () => {
      isCancelled = true
      window.clearInterval(intervalId)
    }
  }, [])

  useEffect(() => {
    if (!jobId || (jobStatus && TERMINAL_JOB_STATUSES.has(jobStatus))) {
      return
    }

    let isCancelled = false

    const pollJob = async () => {
      try {
        const nextJob = await getJob(jobId)
        if (isCancelled) {
          return
        }

        setJob(nextJob)
        setJobRefreshError(null)
      } catch (error) {
        if (isCancelled) {
          return
        }

        setJobRefreshError(
          getApiErrorMessage(error, 'Failed to refresh job status.'),
        )
      }
    }

    void pollJob()
    const intervalId = window.setInterval(() => {
      void pollJob()
    }, POLL_INTERVAL_MS)

    return () => {
      isCancelled = true
      window.clearInterval(intervalId)
    }
  }, [jobId, jobStatus])

  useEffect(() => {
    if (!jobId || jobStatus !== 'succeeded') {
      return
    }

    let isCancelled = false

    const loadArtifacts = async () => {
      setIsArtifactsLoading(true)
        setArtifactError(null)

      const previewResult = await Promise.allSettled([
        getJobPreview(jobId),
      ]).then(([result]) => result)

      if (isCancelled) {
        return
      }

      const nextErrors: string[] = []

      if (previewResult.status === 'fulfilled') {
        setPreview(previewResult.value)
      } else {
        setPreview(null)
        nextErrors.push(
          `Preview: ${getApiErrorMessage(
            previewResult.reason,
            'Failed to load preview.',
          )}`,
        )
      }

      setArtifactError(nextErrors.length > 0 ? nextErrors.join(' ') : null)
      setIsArtifactsLoading(false)
    }

    void loadArtifacts()

    return () => {
      isCancelled = true
    }
  }, [jobId, jobStatus])

  async function handleUpload(file: File) {
    setIsUploading(true)
    setUploadError(null)
    setResumeError(null)
    setJobRefreshError(null)
    setArtifactError(null)
    setIsArtifactsLoading(false)
    setPreview(null)
    setJob(null)

    try {
      const createdJob = await createJob(file)
      setJob(createdJob)
    } catch (error) {
      setUploadError(getApiErrorMessage(error, 'Failed to create job.'))
    } finally {
      setIsUploading(false)
    }
  }

  const frontendStatus: ServiceStatus = {
    checked_at: new Date().toISOString(),
    detail: 'Vite frontend is rendering in the browser.',
    status: 'ok',
  }

  const services = serviceMeta.map((service) => ({
    ...service,
    status:
      service.key === 'frontend'
        ? frontendStatus
        : systemStatus?.services[service.key] ?? {
            checked_at: '',
            detail: 'No system status received yet.',
            status: 'unknown',
          },
  }))

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-top">
          <div>
            <p className="eyebrow">price-tag-vision / minimal job flow</p>
            <h1>Upload a shelf video and watch the job complete.</h1>
            <p className="summary">
              This frontend keeps the MVP deliberately small: upload one video,
              poll the job status, inspect the mock preview, review crop images,
              and download the CSV result.
            </p>
          </div>
          <div className="signature-badge">made by Козырный Бутерброд</div>
        </div>

        <div className="summary-meta">
          <span>API base URL: {API_BASE_URL}</span>
          <span>Polling: {POLL_INTERVAL_MS / 1000}s</span>
        </div>

        <UploadDropzone isUploading={isUploading} onUpload={handleUpload} />
        {uploadError ? <div className="banner banner-error">{uploadError}</div> : null}
        {resumeError ? <div className="banner banner-warning">{resumeError}</div> : null}
      </section>

      {job ? (
        <JobStatusCard
          csvDownloadUrl={csvDownloadUrl}
          isPolling={isPolling}
          job={job}
          refreshError={jobRefreshError}
        />
      ) : (
        <section className="panel empty-panel">
          <p className="eyebrow">Current Job</p>
          <h2>No job yet</h2>
          <p className="empty-copy">
            Upload an MP4 to create a backend job and start polling its status.
          </p>
        </section>
      )}

      <section className="results-grid">
        <section className="panel result-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Preview</p>
              <h2>Result preview</h2>
            </div>
          </div>

          {job?.status === 'succeeded' ? (
            <>
              {artifactError ? (
                <div className="banner banner-warning">{artifactError}</div>
              ) : null}
              {isArtifactsLoading && preview === null ? (
                <p className="empty-copy">Loading preview artifact...</p>
              ) : preview !== null ? (
                <PreviewTable preview={preview} />
              ) : (
                <p className="empty-copy">No preview payload was returned.</p>
              )}
            </>
          ) : (
            <p className="empty-copy">
              Preview appears here after the job reaches <strong>succeeded</strong>.
            </p>
          )}
        </section>
      </section>

      <section className="panel system-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">System Status</p>
            <h2>Foundation block</h2>
          </div>
          <span className={`status-chip status-${systemStatus?.status ?? 'unknown'}`}>
            {systemStatus?.status ?? 'pending'}
          </span>
        </div>

        <p className="system-copy">
          The previous dashboard is preserved as a compact diagnostic block so
          upload issues can still be correlated with backend, worker, storage or
          queue health.
        </p>

        <div className="summary-meta">
          <span>
            Last check:{' '}
            {systemStatus?.checked_at
              ? new Date(systemStatus.checked_at).toLocaleTimeString()
              : 'pending'}
          </span>
          <span>{API_BASE_URL}/api/system/status</span>
        </div>

        {systemError ? <div className="banner banner-warning">{systemError}</div> : null}

        <div className="service-grid">
          {services.map((service) => (
            <article key={service.key} className={`service-card status-${service.status.status}`}>
              <div className="service-head">
                <div>
                  <p className="service-label">{service.label}</p>
                  <h3>{service.status.status}</h3>
                </div>
                <span className="service-dot" aria-hidden="true"></span>
              </div>
              <p className="service-detail">{service.status.detail}</p>
              <div className="service-foot">
                <span>
                  {service.status.checked_at
                    ? new Date(service.status.checked_at).toLocaleTimeString()
                    : 'waiting'}
                </span>
                {service.href ? (
                  <a href={service.href} rel="noreferrer" target="_blank">
                    Open
                  </a>
                ) : (
                  <span>internal</span>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App
