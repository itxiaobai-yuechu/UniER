Name=('ASSISTments2017')
Age=10
N=12
export UNIER_SEED="${UNIER_SEED:-42}"
for name in "${Name[@]}"
do
	mkdir -p ../fin_res/assist2017
	python main.py "$name" $Age "$N" > ../fin_res/assist2017/run.log
	awk -F '\t' 'NF == 2 && $1 ~ /^[0-9]+$/' ../fin_res/assist2017/run.log > ../fin_res/assist2017/recommend.txt
done
