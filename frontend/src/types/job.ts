export type JobStatus =
  | 'uploaded'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | (string & {})

export type Job = {
  id: string
  original_filename: string
  status: JobStatus
  progress: number
  stage: string | null
  message: string | null
  error: string | null
  input_video_key: string
  output_csv_key: string | null
  preview_json_key: string | null
  crop_keys_json: string[] | null
  stats_json: Record<string, unknown> | null
  pipeline_name: string
  pipeline_version: string
  created_at: string
  updated_at: string
}

export type JobCrop = {
  key: string
  url: string
  filename: string
}

export type PreviewPayload = unknown
