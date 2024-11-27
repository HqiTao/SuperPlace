EXTRA_DATASETS_MSLS_10_4 = ['MSLS0amman', 'MSLS6bangkok', 'MSLS11amman', 'MSLS8austin', 'MSLS3amman', 'MSLS4amsterdam', 'MSLS0berlin', 'MSLS6berlin', 'MSLS8goa', 'MSLS13budapest', 'MSLS11amsterdam', 'MSLS15bangkok', 'MSLS13amman', 'MSLS11bangkok', 'MSLS4berlin', 'MSLS8boston', 'MSLS13austin', 'MSLS12boston', 'MSLS7austin', 'MSLS1amsterdam', 'MSLS8amman', 'MSLS6budapest', 'MSLS14bangkok', 'MSLS0bangkok', 'MSLS14goa', 'MSLS12bangkok', 'MSLS11austin', 'MSLS11boston', 'MSLS15amman', 'MSLS7bangkok', 'MSLS9goa', 'MSLS14austin', 'MSLS3amsterdam', 'MSLS8amsterdam', 'MSLS15austin', 'MSLS1berlin', 'MSLS4bangkok', 'MSLS10amsterdam', 'MSLS0austin', 'MSLS0boston', 'MSLS6boston', 'MSLS5amman', 'MSLS11berlin', 'MSLS3austin', 'MSLS1goa', 'MSLS4austin', 'MSLS9bangkok', 'MSLS3goa', 'MSLS10amman', 'MSLS10goa', 'MSLS8bangkok', 'MSLS7boston', 'MSLS10austin', 'MSLS9amman', 'MSLS5budapest', 'MSLS5bangkok', 'MSLS2budapest', 'MSLS6amman', 'MSLS14berlin', 'MSLS9budapest', 'MSLS11budapest', 'MSLS12berlin', 'MSLS1boston', 'MSLS1bangkok', 'MSLS15boston', 'MSLS12goa', 'MSLS15goa', 'MSLS4budapest', 'MSLS0amsterdam', 'MSLS5boston', 'MSLS7goa', 'MSLS1austin', 'MSLS2austin', 'MSLS5goa', 'MSLS3bangkok', 'MSLS15berlin', 'MSLS6amsterdam', 'MSLS13goa', 'MSLS9berlin', 'MSLS0goa', 'MSLS2amsterdam', 'MSLS9amsterdam', 'MSLS3budapest', 'MSLS13bangkok', 'MSLS7berlin', 'MSLS10bangkok', 'MSLS10budapest', 'MSLS10boston', 'MSLS2boston', 'MSLS3berlin', 'MSLS8berlin', 'MSLS7budapest', 'MSLS14amsterdam', 'MSLS1amman', 'MSLS15budapest', 'MSLS11goa', 'MSLS12amsterdam', 'MSLS5austin', 'MSLS4amman', 'MSLS4boston', 'MSLS15amsterdam', 'MSLS10berlin', 'MSLS3boston', 'MSLS7amman', 'MSLS12austin', 'MSLS2goa', 'MSLS7amsterdam', 'MSLS8budapest', 'MSLS9boston', 'MSLS4goa', 'MSLS14budapest', 'MSLS0budapest', 'MSLS13amsterdam', 'MSLS12amman', 'MSLS2bangkok', 'MSLS5berlin', 'MSLS14amman', 'MSLS2berlin', 'MSLS5amsterdam', 'MSLS1budapest', 'MSLS6austin', 'MSLS9austin', 'MSLS14boston', 'MSLS13berlin', 'MSLS6goa', 'MSLS12budapest', 'MSLS13boston', 'MSLS2amman']


import os

target_directory = "/mnt/sda3/2024_Projects/npr/datasets/gsv_cities/Images/"

msls_folders = [folder for folder in os.listdir(target_directory) 
                if os.path.isdir(os.path.join(target_directory, folder)) and folder.startswith("MSLS")]

print("EXTRA_DATASETS_MSLS =", msls_folders)



