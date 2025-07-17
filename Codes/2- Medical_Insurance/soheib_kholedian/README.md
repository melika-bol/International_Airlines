# Medical Inssurance 

in this study I have done research on this [dataset](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset) that is for Medical Inssurance i have tride to find 
what is have realtion on expenses, premium 

## data Anlisis

### 1. data Distribution

the data has two kind of first categoracal and numberic 

![categoracal distribtion](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Distribution.png)

> the age, region are distributed close to evenly between theire classes but the discount_eligibility is 70% no showing the majority of our data are not fitted for a discount

![numbrec distribtion](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Distributions%20of%20Numeric%20Columns.png)

> all the numbric data have the Wrinkles on the roller side even the bmi that has normalize distribution is a littel bet to the wright side 

## 2. outliyer detction 

to find oulyer i have used the IQR rule and the outlier that i have found are 

| Column   | Number of Outliers | Mean of Outliers |
| -------- | ------------------ | ---------------: |
| Age      | 0                  |             None |
| BMI      | 9                  |            49.28 |
| Children | 0                  |             None |
| Expenses | 139                |         42103.95 |
| Premium  | 113                |          1021.56 |

![scatter plot hilelitting the outlier](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Scatter%20Plots%20Highlighting%20Outliers.png)

age dos not have any oultlier in it 

bmi outliyers are not anything to high or wierd consedring the over wieth and fatnes is to hirgh writh now 

in preimum and expences is the place that most of the outlier are and the number are to hiegh our mean are 42103.95, 1021.56 but there are not erorr or made up data
becous cose of insurense is hiegh spethily for peple with spesol seckness 

## 3. reltion study between the featur 

1. Correlation Heatmap of Numeric Features
![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Correlation%20Heatmap%20of%20Numeric%20Features.png)

in this heat map there are some strong linear reltion between premium and expences it's 0.85 and onely other high number is between the age and premium suggesting
that as peple get older thry pay hier for insurenc

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Average%20Premium-Expenses.png)

for or catgoracl data we have some intesting thiges peple with 2 or 3 cheldrian have much hier incurinse cost then peple with 4 or 5 this can suggest that femailly 
more then 4 chial are discount_eligibil for a discount 
> it's just a gess not sure

the other thing is the discount_eligibility peple that have discount pay more then other suggesting that maby peple that have spetiol conetion are more likely
to have disscount

other featur don't have a strond difrence

but to make sure i have look it them more colesly

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Age%20vs%20Expenses%20colored%20by%20Gender.png)

* Both genders show increasing expenses with age; males slightly higher.

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/BMI%20vs%20Premium%20colored%20by%20Region.png)

* Southeast region stands out with higher premiums for similar BMI ranges.

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Insurance%20Premium%20vs%20Expenses.png)

* Distinct diagonal bands; strongest direct relationship; likely reflects pricing tiers.

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Pairplot%20of%20Numeric%20Columns%20colored%20by%20Gender.png)

* Confirms small gender differences; males sometimes higher in expenses and premium.

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Pairplot%20of%20Numeric%20Columns%20colored%20by%20discount_eligibility.png)

* Distinct clustering: "yes" group concentrated at high expenses and premiums; confirms discount\_eligibility as strongest cost driver.

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Scatter%20Plots%20Important%20Features%20vs%20Expenses%20%26%20Premium.png)

* Age shows upward trends; clear cost bands suggest other influencing factors.
* BMI has a mild upward trend; wide scatter shows it’s not the sole driver.
* Number of children: minimal impact.
---
## 2. model selction 

since i have the outliyer and I absoltly didn't wented to delet them and chopping wasn't writh in my opnian i deceded to use model that are robust to outliyer and i used the XGboost and Random Forest 

---
## 3. preprossing 

since the my model didn't requer any normoleztion ar standeredzion i just used lable encoder and one hot encoder for my catogrecal data and split them to test and train 
--- 
## 4. model traing

i train my model by 3 deffrent goale 

1. frist scenario:

i triad to train them to predict bouth premuim and expences at same time

x -> age, gender, bmi, children, discount_eligibility, region

y -> expinces, premuim

2. secand scenario

onley to predict the expences

x -> age, gender, bmi, children, discount_eligibility, region, premuim

y -> expinces

3. 3rd scenario

onley to predict the premuim

x -> age, gender, bmi, children, discount_eligibility, region, expences

y -> premuim

---

5.valedtion

i use shap to see the reltion between x and y and conferm my finding in data anlisi and find any unlinyer realtion

first senreio

Feature	MAE	MSE	RMSE	R2
expenses	2524.890146	21303382.92	4615.558787	0.862779023
premium	43.75510628	6846.682495	82.74468258	0.926017562


