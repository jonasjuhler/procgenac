
for file in jobscripts/*.sh
do
  bsub < $file
done