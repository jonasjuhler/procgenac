
for file in /zhome/04/d/118529/projects/procgenac/hpcfiles/gridsearch/jobscripts/*.sh
do
  bsub < $file
done