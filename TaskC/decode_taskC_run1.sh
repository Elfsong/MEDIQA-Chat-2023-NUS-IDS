
if [ "$#" -ne 1 ]
then
    echo "input csv filepath expected"
else
	fn="taskC_NUSIDS_run1.csv"
	echo "writing output to "$fn
	python TaskCRun1.py $1 $fn
fi
