import type { JobCrop } from '../types/job'

type CropGalleryProps = {
  crops: JobCrop[]
}

export function CropGallery({ crops }: CropGalleryProps) {
  if (crops.length === 0) {
    return <p className="empty-copy">No crop images were returned for this job.</p>
  }

  return (
    <div className="crop-grid">
      {crops.map((crop) => (
        <figure key={crop.key} className="crop-card">
          <img
            alt={crop.filename}
            className="crop-image"
            loading="lazy"
            src={crop.url}
          />
          <figcaption>{crop.filename}</figcaption>
        </figure>
      ))}
    </div>
  )
}
