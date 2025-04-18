
mileageList = ["2500", "5000", "10000", "20000", "30000", "40000", "50000",
                       "60000", "70000", "80000", "90000", "100000",
                       "125000", "150000", "175000", "200000"]
for mileage_from, mileage_to in zip(mileageList, mileageList[1:]):  # Loop over adjacent elements
    print(mileage_from, mileage_to)