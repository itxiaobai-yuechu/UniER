Name=('ASSISTments2017')
Age=10
N=12
for name in "${Name[@]}"
do
	python main.py "$name" $Age "$N" > ../fin_res/gpu-ER-TGA-$name-Age$Age-N$N-Psize30-Epsilon05-1-10.txt &
done