import { useState } from 'react'
import { useDropzone } from 'react-dropzone'

type UploadDropzoneProps = {
  isUploading: boolean
  onUpload: (file: File) => Promise<void> | void
}

export function UploadDropzone({
  isUploading,
  onUpload,
}: UploadDropzoneProps) {
  const [dropError, setDropError] = useState<string | null>(null)

  const { getInputProps, getRootProps, isDragActive, isDragReject } =
    useDropzone({
      accept: {
        'video/mp4': ['.mp4'],
      },
      disabled: isUploading,
      maxFiles: 1,
      multiple: false,
      onDropAccepted: (acceptedFiles) => {
        const [file] = acceptedFiles
        if (!file) {
          return
        }

        setDropError(null)
        void onUpload(file)
      },
      onDropRejected: (rejections) => {
        const [firstRejection] = rejections
        const [firstError] = firstRejection?.errors ?? []

        setDropError(firstError?.message ?? 'Only one MP4 file is supported.')
      },
    })

  const dropzoneClassName = [
    'dropzone',
    isDragActive ? 'dropzone-active' : '',
    isDragReject ? 'dropzone-reject' : '',
    isUploading ? 'dropzone-disabled' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className="dropzone-shell">
      <div {...getRootProps({ className: dropzoneClassName })}>
        <input {...getInputProps()} />
        <p className="dropzone-title">
          {isUploading ? 'Uploading video...' : 'Drop an MP4 here or click to browse'}
        </p>
        <p className="dropzone-copy">
          The frontend creates a job immediately, then polls the backend every
          1.5 seconds until the result is ready.
        </p>
      </div>
      {dropError ? <div className="banner banner-error">{dropError}</div> : null}
    </div>
  )
}
