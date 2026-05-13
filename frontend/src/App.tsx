import { useEffect, useMemo, useState } from 'react'
import './App.css'

type ServiceStatus = {
  status: string
  detail: string
  checked_at: string
}

type StatusPayload = {
  service: string
  status: string
  checked_at: string
  services: Record<string, ServiceStatus>
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

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
  const [status, setStatus] = useState<StatusPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    const loadStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/system/status`)
        const payload = (await response.json()) as StatusPayload

        if (!active) {
          return
        }

        setStatus(payload)
        setError(response.ok ? null : 'One or more services are still unhealthy.')
      } catch (requestError) {
        if (!active) {
          return
        }

        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Status request failed.',
        )
      }
    }

    loadStatus()
    const intervalId = window.setInterval(loadStatus, 5000)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  const services = useMemo(() => {
    const frontendStatus: ServiceStatus = {
      status: 'ok',
      detail: 'Vite frontend is responding in the browser.',
      checked_at: new Date().toISOString(),
    }

    return serviceMeta.map((service) => ({
      ...service,
      status:
        service.key === 'frontend'
          ? frontendStatus
          : status?.services[service.key] ?? {
              status: 'unknown',
              detail: 'No status received yet.',
              checked_at: '',
            },
    }))
  }, [status])

  const headline =
    status?.status === 'ok'
      ? 'Foundation is up'
      : 'Foundation is still converging'

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <p className="eyebrow">price-tag-vision / docker foundation</p>
        <h1>{headline}</h1>
        <p className="summary">
          Minimal baseline for the hackathon stack: backend, worker, vision
          service, queue, storage and database are wired together and report
          their health from one place.
        </p>
        <div className="summary-meta">
          <span>API: {API_BASE_URL}</span>
          <span>
            Last check:{' '}
            {status?.checked_at
              ? new Date(status.checked_at).toLocaleTimeString()
              : 'pending'}
          </span>
        </div>
        {error ? <div className="status-banner warning">{error}</div> : null}
      </section>

      <section className="services-grid">
        {services.map((service) => (
          <article
            key={service.key}
            className={`service-card status-${service.status.status}`}
          >
            <div className="service-head">
              <div>
                <p className="service-label">{service.label}</p>
                <h2>{service.status.status}</h2>
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
                <a href={service.href} target="_blank" rel="noreferrer">
                  Open
                </a>
              ) : (
                <span>internal</span>
              )}
            </div>
          </article>
        ))}
      </section>

      <section className="endpoints-panel">
        <div>
          <p className="eyebrow">useful endpoints</p>
          <ul className="endpoint-list">
            <li>
              <code>{API_BASE_URL}/health</code>
            </li>
            <li>
              <code>{API_BASE_URL}/api/system/status</code>
            </li>
            <li>
              <code>http://localhost:9001/health</code>
            </li>
            <li>
              <code>http://localhost:9002</code>
            </li>
          </ul>
        </div>
        <div>
          <p className="eyebrow">next step</p>
          <p className="next-step-copy">
            This baseline is intentionally narrow: after the stack is healthy,
            you can add job lifecycle, upload flow and the real CV/OCR pipeline
            without rewriting the container topology.
          </p>
        </div>
      </section>
    </main>
  )
}

export default App
