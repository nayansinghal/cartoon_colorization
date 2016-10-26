# images located in $1. The script WILL overwrite.
# put batch-cart2sketch.scm in ~/.gimp/scripts
FOLDER=$1
gimp -i -b '(batch-cart2sketch "'$1'/*.bmp" '$2')' -b '(gimp-quit 0)'
