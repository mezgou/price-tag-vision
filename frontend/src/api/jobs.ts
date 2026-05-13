import { API_BASE_URL, apiClient } from './client'
import type { Job, JobCrop, PreviewPayload } from '../types/job'

export async function createJob(file: File): Promise<Job> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post<Job>('/api/jobs', formData)
  return response.data
}

export async function getJob(jobId: string): Promise<Job> {
  const response = await apiClient.get<Job>(`/api/jobs/${jobId}`)
  return response.data
}

export async function getJobPreview(jobId: string): Promise<PreviewPayload> {
  const response = await apiClient.get<PreviewPayload>(`/api/jobs/${jobId}/preview`)
  return response.data
}

export async function getJobCrops(jobId: string): Promise<JobCrop[]> {
  const response = await apiClient.get<JobCrop[]>(`/api/jobs/${jobId}/crops`)
  return response.data
}

export function getCsvDownloadUrl(jobId: string): string {
  const trimmedBaseUrl = API_BASE_URL.replace(/\/+$/, '')
  return `${trimmedBaseUrl}/api/jobs/${jobId}/download/csv`
}
