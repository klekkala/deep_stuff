for f in ../fin/rgb/*
do
	convert $f -resize 224x224! $f

done
