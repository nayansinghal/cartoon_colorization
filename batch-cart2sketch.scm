(define (batch-cart2sketch pattern thresh) (

	let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
		(let* ((filename (car filelist))
			(image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
			(drawable (car (gimp-image-get-active-layer image))))

			(plug-in-cartoon RUN-NONINTERACTIVE image drawable 7.0 0.2)
			(gimp-threshold drawable thresh 255)	
			(gimp-levels drawable HISTOGRAM-VALUE 0 30 2 0 255)
			(gimp-levels drawable HISTOGRAM-VALUE 0 30 2 0 255)
			(plug-in-antialias TRUE image drawable)

			(gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
			(gimp-image-delete image)
		)
		(set! filelist (cdr filelist)))
))
