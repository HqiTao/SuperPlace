# https://github.com/amaralibey/gsv-cities

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

# GSV-Cities
TRAIN_CITIES = ['Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL',
                'Rome', 'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix',
                'TRT', 'Boston', 'Lisbon', 'Medellin', 'Minneapolis',
                'PRG', 'WashingtonDC', 'Brussels', 'London',
                'Melbourne', 'Osaka', 'PRS',]

# SuperPlace
EXTRA_DATASETS_MSLS = ['MSLS0austin', 'MSLS0bangkok', 'MSLS0berlin', 'MSLS0boston', 'MSLS0budapest', 'MSLS0helsinki', 'MSLS0melbourne', 
               'MSLS0moscow', 'MSLS0ottawa', 'MSLS0paris', 'MSLS0phoenix', 'MSLS0saopaulo', 'MSLS0tokyo', 'MSLS0toronto', 
               'MSLS10austin', 'MSLS10bangkok', 'MSLS10berlin', 'MSLS10boston', 'MSLS10budapest', 'MSLS10helsinki', 
               'MSLS10melbourne', 'MSLS10moscow', 'MSLS10ottawa', 'MSLS10paris', 'MSLS10phoenix', 'MSLS10saopaulo', 'MSLS10tokyo', 
               'MSLS11austin', 'MSLS11bangkok', 'MSLS11berlin', 'MSLS11boston', 'MSLS11budapest', 'MSLS11helsinki', 'MSLS11melbourne', 
               'MSLS11moscow', 'MSLS11ottawa', 'MSLS11paris', 'MSLS11phoenix', 'MSLS11saopaulo', 'MSLS11tokyo', 
               'MSLS12austin', 'MSLS12bangkok', 'MSLS12berlin', 'MSLS12boston', 'MSLS12budapest', 'MSLS12helsinki', 
               'MSLS12melbourne', 'MSLS12moscow', 'MSLS12ottawa', 'MSLS12paris', 'MSLS12phoenix', 'MSLS12saopaulo', 'MSLS12tokyo', 
               'MSLS13austin', 'MSLS13bangkok', 'MSLS13berlin', 'MSLS13boston', 'MSLS13budapest', 'MSLS13helsinki', 'MSLS13melbourne', 
               'MSLS13moscow', 'MSLS13ottawa', 'MSLS13paris', 'MSLS13phoenix', 'MSLS13saopaulo', 'MSLS13tokyo', 
               'MSLS14austin', 'MSLS14bangkok', 'MSLS14berlin', 'MSLS14boston', 'MSLS14budapest', 'MSLS14helsinki', 
               'MSLS14melbourne', 'MSLS14moscow', 'MSLS14ottawa', 'MSLS14paris', 'MSLS14phoenix', 'MSLS14saopaulo', 
               'MSLS14tokyo', 'MSLS14toronto', 'MSLS15austin', 'MSLS15bangkok', 'MSLS15berlin', 'MSLS15boston', 'MSLS15budapest', 
               'MSLS15helsinki', 'MSLS15melbourne', 'MSLS15moscow', 'MSLS15ottawa', 'MSLS15paris', 'MSLS15phoenix', 'MSLS15saopaulo', 'MSLS15tokyo', 'MSLS15toronto', 
               'MSLS16austin', 'MSLS16bangkok', 'MSLS16berlin', 'MSLS16boston', 'MSLS16budapest', 'MSLS16helsinki', 
               'MSLS16melbourne', 'MSLS16moscow', 'MSLS16ottawa', 'MSLS16paris', 'MSLS16phoenix', 'MSLS16saopaulo', 'MSLS16tokyo', 'MSLS16toronto', 
               'MSLS17austin', 'MSLS17bangkok', 'MSLS17berlin', 'MSLS17boston', 'MSLS17budapest', 'MSLS17helsinki', 'MSLS17melbourne', 'MSLS17moscow', 
               'MSLS17ottawa', 'MSLS17paris', 'MSLS17phoenix', 'MSLS17saopaulo', 'MSLS17tokyo', 
               'MSLS18austin', 'MSLS18bangkok', 'MSLS18berlin', 'MSLS18boston', 'MSLS18budapest', 'MSLS18helsinki', 'MSLS18melbourne', 
               'MSLS18moscow', 'MSLS18ottawa', 'MSLS18paris', 'MSLS18phoenix', 'MSLS18saopaulo', 'MSLS18tokyo', 
               'MSLS19austin', 'MSLS19bangkok', 'MSLS19berlin', 'MSLS19boston', 'MSLS19budapest', 'MSLS19helsinki', 'MSLS19melbourne', 
               'MSLS19moscow', 'MSLS19ottawa', 'MSLS19paris', 'MSLS19phoenix', 'MSLS19saopaulo', 'MSLS19tokyo', 'MSLS1austin', 
               'MSLS1bangkok', 'MSLS1berlin', 'MSLS1boston', 'MSLS1budapest', 'MSLS1helsinki', 'MSLS1melbourne', 'MSLS1moscow', 
               'MSLS1ottawa', 'MSLS1paris', 'MSLS1phoenix', 'MSLS1saopaulo', 'MSLS1tokyo', 'MSLS1toronto', 
               'MSLS20austin', 'MSLS20bangkok', 'MSLS20berlin', 'MSLS20boston', 'MSLS20budapest', 'MSLS20helsinki', 'MSLS20melbourne', 
               'MSLS20moscow', 'MSLS20ottawa', 'MSLS20paris', 'MSLS20phoenix', 'MSLS20saopaulo', 'MSLS20tokyo', 'MSLS20toronto', 
               'MSLS21austin', 'MSLS21bangkok', 'MSLS21berlin', 'MSLS21boston', 'MSLS21budapest', 'MSLS21helsinki', 'MSLS21melbourne', 
               'MSLS21moscow', 'MSLS21ottawa', 'MSLS21paris', 'MSLS21phoenix', 'MSLS21saopaulo', 'MSLS21tokyo', 'MSLS22austin', 
               'MSLS22bangkok', 'MSLS22berlin', 'MSLS22boston', 'MSLS22budapest', 'MSLS22helsinki', 'MSLS22melbourne', 'MSLS22moscow', 
               'MSLS22ottawa', 'MSLS22paris', 'MSLS22phoenix', 'MSLS22saopaulo', 'MSLS22tokyo', 
               'MSLS23austin', 'MSLS23bangkok', 'MSLS23berlin', 'MSLS23boston', 'MSLS23budapest', 'MSLS23helsinki', 'MSLS23melbourne', 
               'MSLS23moscow', 'MSLS23ottawa', 'MSLS23paris', 'MSLS23phoenix', 'MSLS23saopaulo', 'MSLS23tokyo', 'MSLS23toronto', 
               'MSLS24austin', 'MSLS24bangkok', 'MSLS24berlin', 'MSLS24boston', 'MSLS24budapest', 'MSLS24helsinki', 'MSLS24melbourne', 
               'MSLS24moscow', 'MSLS24ottawa', 'MSLS24paris', 'MSLS24phoenix', 'MSLS24saopaulo', 'MSLS24tokyo', 
               'MSLS2austin', 'MSLS2bangkok', 'MSLS2berlin', 'MSLS2boston', 'MSLS2budapest', 'MSLS2helsinki', 'MSLS2melbourne', 'MSLS2moscow', 
               'MSLS2ottawa', 'MSLS2paris', 'MSLS2phoenix', 'MSLS2saopaulo', 'MSLS2tokyo', 'MSLS2toronto', 'MSLS3austin', 'MSLS3bangkok', 
               'MSLS3berlin', 'MSLS3boston', 'MSLS3budapest', 'MSLS3helsinki', 'MSLS3melbourne', 'MSLS3moscow', 'MSLS3ottawa', 'MSLS3paris', 
               'MSLS3phoenix', 'MSLS3saopaulo', 'MSLS3tokyo', 'MSLS3toronto', 
               'MSLS4austin', 'MSLS4bangkok', 'MSLS4berlin', 'MSLS4boston', 'MSLS4budapest', 'MSLS4helsinki', 'MSLS4melbourne', 'MSLS4moscow', 
               'MSLS4ottawa', 'MSLS4paris', 'MSLS4phoenix', 'MSLS4saopaulo', 'MSLS4tokyo', 'MSLS4toronto', 
               'MSLS5austin', 'MSLS5bangkok', 'MSLS5berlin', 'MSLS5boston', 'MSLS5budapest', 'MSLS5helsinki', 'MSLS5melbourne', 
               'MSLS5moscow', 'MSLS5ottawa', 'MSLS5paris', 'MSLS5phoenix', 'MSLS5saopaulo', 'MSLS5tokyo', 
               'MSLS6austin', 'MSLS6bangkok', 'MSLS6berlin', 'MSLS6boston', 'MSLS6budapest', 'MSLS6helsinki', 'MSLS6melbourne', 'MSLS6moscow', 
               'MSLS6ottawa', 'MSLS6paris', 'MSLS6phoenix', 'MSLS6saopaulo', 'MSLS6tokyo', 
               'MSLS7austin', 'MSLS7bangkok', 'MSLS7berlin', 'MSLS7boston', 'MSLS7budapest', 'MSLS7helsinki', 'MSLS7melbourne', 'MSLS7moscow', 
               'MSLS7ottawa', 'MSLS7paris', 'MSLS7phoenix', 'MSLS7saopaulo', 'MSLS7tokyo', 'MSLS7toronto', 
               'MSLS8austin', 'MSLS8bangkok', 'MSLS8berlin', 'MSLS8boston', 'MSLS8budapest', 'MSLS8helsinki', 'MSLS8melbourne', 'MSLS8moscow', 
               'MSLS8ottawa', 'MSLS8paris', 'MSLS8phoenix', 'MSLS8saopaulo', 'MSLS8tokyo', 
               'MSLS9austin', 'MSLS9bangkok', 'MSLS9berlin', 'MSLS9boston', 'MSLS9budapest', 'MSLS9helsinki', 'MSLS9melbourne', 'MSLS9moscow', 'MSLS9ottawa', 
               'MSLS9paris', 'MSLS9phoenix', 'MSLS9saopaulo', 'MSLS9tokyo', 'MSLS9toronto', 'MSLS9trondheim']

# EXTRA_DATASETS_MSLS = ['MSLS0amman', 'MSLS6bangkok', 'MSLS32boston', 'MSLS5zurich', 'MSLS27melbourne', 'MSLS16london', 'MSLS31london', 'MSLS35phoenix', 'MSLS32trondheim', 'MSLS32austin', 'MSLS29trondheim', 'MSLS1paris', 'MSLS15manila', 'MSLS11amman', 'MSLS31ottawa', 'MSLS27amman', 'MSLS22trondheim', 'MSLS19tokyo', 'MSLS2zurich', 'MSLS3moscow', 'MSLS8austin', 'MSLS3manila', 'MSLS2tokyo', 'MSLS24boston', 'MSLS20berlin', 'MSLS25trondheim', 'MSLS6london', 'MSLS19london', 'MSLS3amman', 'MSLS18london', 'MSLS2phoenix', 'MSLS0paris', 'MSLS8tokyo', 'MSLS26saopaulo', 'MSLS24ottawa', 'MSLS13moscow', 'MSLS17bangkok', 'MSLS12saopaulo', 'MSLS35ottawa', 'MSLS33phoenix', 'MSLS18goa', 'MSLS26berlin', 'MSLS33boston', 'MSLS35manila', 'MSLS29phoenix', 'MSLS24london', 'MSLS27toronto', 'MSLS23ottawa', 'MSLS32moscow', 'MSLS35paris', 'MSLS1tokyo', 'MSLS4amsterdam', 'MSLS27manila', 'MSLS0berlin', 'MSLS15melbourne', 'MSLS6berlin', 'MSLS26moscow', 'MSLS28manila', 'MSLS17goa', 'MSLS5saopaulo', 'MSLS18phoenix', 'MSLS8goa', 'MSLS16tokyo', 'MSLS13budapest', 'MSLS20saopaulo', 'MSLS11amsterdam', 'MSLS28tokyo', 'MSLS21manila', 'MSLS25manila', 'MSLS22nairobi', 'MSLS21bangkok', 'MSLS4manila', 'MSLS26nairobi', 'MSLS14helsinki', 'MSLS32amsterdam', 'MSLS31berlin', 'MSLS21goa', 'MSLS15bangkok', 'MSLS30boston', 'MSLS34moscow', 'MSLS28goa', 'MSLS24melbourne', 'MSLS27tokyo', 'MSLS32toronto', 'MSLS21ottawa', 'MSLS7melbourne', 'MSLS13amman', 'MSLS14moscow', 'MSLS16manila', 'MSLS20amsterdam', 'MSLS15moscow', 'MSLS17saopaulo', 'MSLS7manila', 'MSLS22amman', 'MSLS27zurich', 'MSLS8ottawa', 'MSLS22toronto', 'MSLS11bangkok', 'MSLS20trondheim', 'MSLS31helsinki', 'MSLS8toronto', 'MSLS34manila', 'MSLS4berlin', 'MSLS19boston', 'MSLS20melbourne', 'MSLS1melbourne', 'MSLS25boston', 'MSLS31trondheim', 'MSLS10london', 'MSLS11phoenix', 'MSLS8boston', 'MSLS17paris', 'MSLS23amsterdam', 'MSLS14paris', 'MSLS16melbourne', 'MSLS14london', 'MSLS27nairobi', 'MSLS1nairobi', 'MSLS18paris', 'MSLS13austin', 'MSLS24trondheim', 'MSLS4moscow', 'MSLS17boston', 'MSLS33ottawa', 'MSLS25moscow', 'MSLS22moscow', 'MSLS28berlin', 'MSLS16toronto', 'MSLS12boston', 'MSLS1moscow', 'MSLS8nairobi', 'MSLS27trondheim', 'MSLS30bangkok', 'MSLS9paris', 'MSLS13phoenix', 'MSLS34phoenix', 'MSLS18berlin', 'MSLS20zurich', 'MSLS32ottawa', 'MSLS17melbourne', 'MSLS17budapest', 'MSLS5trondheim', 'MSLS7austin', 'MSLS4saopaulo', 'MSLS30toronto', 'MSLS1amsterdam', 'MSLS29moscow', 'MSLS8amman', 'MSLS6budapest', 'MSLS14bangkok', 'MSLS22tokyo', 'MSLS15trondheim', 'MSLS33berlin', 'MSLS1zurich', 'MSLS16moscow', 'MSLS0bangkok', 'MSLS12moscow', 'MSLS14goa', 'MSLS12bangkok', 'MSLS25toronto', 'MSLS11austin', 'MSLS18austin', 'MSLS11boston', 'MSLS6tokyo', 'MSLS33tokyo', 'MSLS30tokyo', 'MSLS13melbourne', 'MSLS19manila', 'MSLS16zurich', 'MSLS19saopaulo', 'MSLS17ottawa', 'MSLS8zurich', 'MSLS26boston', 'MSLS13paris', 'MSLS28amman', 'MSLS8moscow', 'MSLS0trondheim', 'MSLS33zurich', 'MSLS20manila', 'MSLS15amman', 'MSLS34paris', 'MSLS7bangkok', 'MSLS31amman', 'MSLS32manila', 'MSLS9goa', 'MSLS24amman', 'MSLS23tokyo', 'MSLS23phoenix', 'MSLS14austin', 'MSLS3amsterdam', 'MSLS27bangkok', 'MSLS18toronto', 'MSLS6zurich', 'MSLS30london', 'MSLS8amsterdam', 'MSLS11nairobi', 'MSLS0london', 'MSLS20paris', 'MSLS21tokyo', 'MSLS24zurich', 'MSLS30melbourne', 'MSLS16austin', 'MSLS33budapest', 'MSLS28austin', 'MSLS6helsinki', 'MSLS23toronto', 'MSLS31saopaulo', 'MSLS19melbourne', 'MSLS5moscow', 'MSLS29amsterdam', 'MSLS15austin', 'MSLS28trondheim', 'MSLS33austin', 'MSLS27berlin', 'MSLS21amsterdam', 'MSLS20bangkok', 'MSLS26manila', 'MSLS29london', 'MSLS1berlin', 'MSLS6manila', 'MSLS25goa', 'MSLS5helsinki', 'MSLS34helsinki', 'MSLS34zurich', 'MSLS15paris', 'MSLS30trondheim', 'MSLS27boston', 'MSLS4bangkok', 'MSLS33amsterdam', 'MSLS22zurich', 'MSLS17manila', 'MSLS23austin', 'MSLS25saopaulo', 'MSLS30moscow', 'MSLS13helsinki', 'MSLS10amsterdam', 'MSLS30nairobi', 'MSLS17toronto', 'MSLS0austin', 'MSLS9nairobi', 'MSLS0boston', 'MSLS28london', 'MSLS6boston', 'MSLS29nairobi', 'MSLS8paris', 'MSLS7moscow', 'MSLS29goa', 'MSLS10trondheim', 'MSLS32paris', 'MSLS18bangkok', 'MSLS21amman', 'MSLS16amsterdam', 'MSLS24helsinki', 'MSLS35zurich', 'MSLS24nairobi', 'MSLS18nairobi', 'MSLS24bangkok', 'MSLS6ottawa', 'MSLS19bangkok', 'MSLS5amman', 'MSLS11berlin', 'MSLS10zurich', 'MSLS9london', 'MSLS3austin', 'MSLS20boston', 'MSLS19amman', 'MSLS23budapest', 'MSLS4tokyo', 'MSLS30helsinki', 'MSLS31austin', 'MSLS1goa', 'MSLS4austin', 'MSLS12nairobi', 'MSLS14phoenix', 'MSLS2paris', 'MSLS0ottawa', 'MSLS14manila', 'MSLS18manila', 'MSLS28melbourne', 'MSLS13tokyo', 'MSLS26zurich', 'MSLS9bangkok', 'MSLS21boston', 'MSLS9phoenix', 'MSLS26amsterdam', 'MSLS11toronto', 'MSLS3goa', 'MSLS5melbourne', 'MSLS23saopaulo', 'MSLS26paris', 'MSLS10amman', 'MSLS21trondheim', 'MSLS35helsinki', 'MSLS3ottawa', 'MSLS17zurich', 'MSLS14nairobi', 'MSLS12zurich', 'MSLS16budapest', 'MSLS16phoenix', 'MSLS31amsterdam', 'MSLS19austin', 'MSLS29boston', 'MSLS14melbourne', 'MSLS1toronto', 'MSLS34bangkok', 'MSLS10goa', 'MSLS5ottawa', 'MSLS26tokyo', 'MSLS17moscow', 'MSLS7ottawa', 'MSLS32melbourne', 'MSLS35moscow', 'MSLS8bangkok', 'MSLS19toronto', 'MSLS11zurich', 'MSLS9manila', 'MSLS32bangkok', 'MSLS30saopaulo', 'MSLS10moscow', 'MSLS7boston', 'MSLS18moscow', 'MSLS11manila', 'MSLS3nairobi', 'MSLS10austin', 'MSLS33nairobi', 'MSLS18budapest', 'MSLS2ottawa', 'MSLS9amman', 'MSLS27paris', 'MSLS28amsterdam', 'MSLS8phoenix', 'MSLS10tokyo', 'MSLS5budapest', 'MSLS7phoenix', 'MSLS5bangkok', 'MSLS19trondheim', 'MSLS32goa', 'MSLS0toronto', 'MSLS8manila', 'MSLS23moscow', 'MSLS16berlin', 'MSLS33bangkok', 'MSLS23amman', 'MSLS3london', 'MSLS2budapest', 'MSLS20austin', 'MSLS22phoenix', 'MSLS13ottawa', 'MSLS21helsinki', 'MSLS23melbourne', 'MSLS6amman', 'MSLS2helsinki', 'MSLS26melbourne', 'MSLS19nairobi', 'MSLS3toronto', 'MSLS13toronto', 'MSLS32budapest', 'MSLS20nairobi', 'MSLS12manila', 'MSLS32tokyo', 'MSLS22budapest', 'MSLS25austin', 'MSLS18melbourne', 'MSLS33saopaulo', 'MSLS25london', 'MSLS20phoenix', 'MSLS14berlin', 'MSLS9budapest', 'MSLS35nairobi', 'MSLS11budapest', 'MSLS12berlin', 'MSLS34austin', 'MSLS29manila', 'MSLS1boston', 'MSLS1bangkok', 'MSLS33moscow', 'MSLS15boston', 'MSLS11moscow', 'MSLS28paris', 'MSLS23zurich', 'MSLS31manila', 'MSLS12helsinki', 'MSLS35budapest', 'MSLS30ottawa', 'MSLS24amsterdam', 'MSLS26bangkok', 'MSLS3helsinki', 'MSLS7paris', 'MSLS19ottawa', 'MSLS12trondheim', 'MSLS35london', 'MSLS25paris', 'MSLS3phoenix', 'MSLS35boston', 'MSLS20helsinki', 'MSLS34goa', 'MSLS14zurich', 'MSLS12goa', 'MSLS25ottawa', 'MSLS20moscow', 'MSLS15goa', 'MSLS29helsinki', 'MSLS18amsterdam', 'MSLS4budapest', 'MSLS23berlin', 'MSLS31nairobi', 'MSLS23goa', 'MSLS23london', 'MSLS0amsterdam', 'MSLS27moscow', 'MSLS5boston', 'MSLS2nairobi', 'MSLS22boston', 'MSLS29melbourne', 'MSLS27phoenix', 'MSLS29toronto', 'MSLS27ottawa', 'MSLS35melbourne', 'MSLS7goa', 'MSLS15zurich', 'MSLS7toronto', 'MSLS21berlin', 'MSLS9saopaulo', 'MSLS34budapest', 'MSLS9tokyo', 'MSLS26helsinki', 'MSLS6paris', 'MSLS1austin', 'MSLS33amman', 'MSLS26goa', 'MSLS34saopaulo', 'MSLS16saopaulo', 'MSLS0nairobi', 'MSLS4paris', 'MSLS29zurich', 'MSLS2austin', 'MSLS2saopaulo', 'MSLS7london', 'MSLS28ottawa', 'MSLS13zurich', 'MSLS0manila', 'MSLS5goa', 'MSLS7trondheim', 'MSLS21budapest', 'MSLS28toronto', 'MSLS26austin', 'MSLS28nairobi', 'MSLS29tokyo', 'MSLS31goa', 'MSLS3bangkok', 'MSLS19moscow', 'MSLS19helsinki', 'MSLS15berlin', 'MSLS29saopaulo', 'MSLS31toronto', 'MSLS0helsinki', 'MSLS16helsinki', 'MSLS35amsterdam', 'MSLS15saopaulo', 'MSLS32zurich', 'MSLS34amman', 'MSLS6saopaulo', 'MSLS25phoenix', 'MSLS4ottawa', 'MSLS10nairobi', 'MSLS15phoenix', 'MSLS23paris', 'MSLS20ottawa', 'MSLS23boston', 'MSLS20budapest', 'MSLS16nairobi', 'MSLS0melbourne', 'MSLS11ottawa', 'MSLS6amsterdam', 'MSLS21melbourne', 'MSLS21paris', 'MSLS13goa', 'MSLS35austin', 'MSLS9berlin', 'MSLS3tokyo', 'MSLS1manila', 'MSLS22saopaulo', 'MSLS17helsinki', 'MSLS0goa', 'MSLS2amsterdam', 'MSLS2moscow', 'MSLS1saopaulo', 'MSLS27budapest', 'MSLS22london', 'MSLS12london', 'MSLS30budapest', 'MSLS27austin', 'MSLS29ottawa', 'MSLS9amsterdam', 'MSLS19budapest', 'MSLS24saopaulo', 'MSLS9trondheim', 'MSLS11saopaulo', 'MSLS25melbourne', 'MSLS10toronto', 'MSLS25amman', 'MSLS33london', 'MSLS2trondheim', 'MSLS8helsinki', 'MSLS22helsinki', 'MSLS16ottawa', 'MSLS23nairobi', 'MSLS27saopaulo', 'MSLS28zurich', 'MSLS24goa', 'MSLS7tokyo', 'MSLS14tokyo', 'MSLS11melbourne', 'MSLS31moscow', 'MSLS14toronto', 'MSLS32amman', 'MSLS3zurich', 'MSLS3budapest', 'MSLS16boston', 'MSLS13bangkok', 'MSLS24tokyo', 'MSLS31paris', 'MSLS7berlin', 'MSLS19phoenix', 'MSLS10bangkok', 'MSLS25bangkok', 'MSLS10budapest', 'MSLS22bangkok', 'MSLS2toronto', 'MSLS33trondheim', 'MSLS10boston', 'MSLS13trondheim', 'MSLS20amman', 'MSLS4zurich', 'MSLS34melbourne', 'MSLS7nairobi', 'MSLS26phoenix', 'MSLS10helsinki', 'MSLS18boston', 'MSLS20tokyo', 'MSLS3trondheim', 'MSLS35saopaulo', 'MSLS22goa', 'MSLS34boston', 'MSLS18tokyo', 'MSLS21london', 'MSLS32nairobi', 'MSLS8saopaulo', 'MSLS4nairobi', 'MSLS2boston', 'MSLS6phoenix', 'MSLS20london', 'MSLS4trondheim', 'MSLS33manila', 'MSLS3berlin', 'MSLS17berlin', 'MSLS8berlin', 'MSLS18helsinki', 'MSLS1london', 'MSLS34trondheim', 'MSLS4helsinki', 'MSLS4london', 'MSLS30amsterdam', 'MSLS7budapest', 'MSLS5tokyo', 'MSLS10ottawa', 'MSLS29bangkok', 'MSLS14amsterdam', 'MSLS0phoenix', 'MSLS19goa', 'MSLS22melbourne', 'MSLS34toronto', 'MSLS6nairobi', 'MSLS1amman', 'MSLS11tokyo', 'MSLS17phoenix', 'MSLS19zurich', 'MSLS14ottawa', 'MSLS16amman', 'MSLS19berlin', 'MSLS0moscow', 'MSLS24berlin', 'MSLS9toronto', 'MSLS6melbourne', 'MSLS23bangkok', 'MSLS31zurich', 'MSLS9helsinki', 'MSLS11trondheim', 'MSLS8trondheim', 'MSLS15budapest', 'MSLS19amsterdam', 'MSLS16trondheim', 'MSLS35trondheim', 'MSLS1helsinki', 'MSLS26toronto', 'MSLS11goa', 'MSLS17nairobi', 'MSLS1trondheim', 'MSLS34tokyo', 'MSLS4toronto', 'MSLS25budapest', 'MSLS12amsterdam', 'MSLS5austin', 'MSLS2melbourne', 'MSLS28budapest', 'MSLS4amman', 'MSLS24austin', 'MSLS6toronto', 'MSLS34berlin', 'MSLS32saopaulo', 'MSLS4boston', 'MSLS25tokyo', 'MSLS13london', 'MSLS3saopaulo', 'MSLS31melbourne', 'MSLS15amsterdam', 'MSLS30amman', 'MSLS28boston', 'MSLS30phoenix', 'MSLS17london', 'MSLS10berlin', 'MSLS15nairobi', 'MSLS27amsterdam', 'MSLS35berlin', 'MSLS28helsinki', 'MSLS18saopaulo', 'MSLS22austin', 'MSLS1phoenix', 'MSLS20toronto', 'MSLS35amman', 'MSLS25amsterdam', 'MSLS24paris', 'MSLS11paris', 'MSLS3boston', 'MSLS21saopaulo', 'MSLS7amman', 'MSLS13manila', 'MSLS8melbourne', 'MSLS12austin', 'MSLS30austin', 'MSLS32london', 'MSLS30berlin', 'MSLS7saopaulo', 'MSLS24phoenix', 'MSLS2goa', 'MSLS14saopaulo', 'MSLS17trondheim', 'MSLS16goa', 'MSLS7amsterdam', 'MSLS27london', 'MSLS10saopaulo', 'MSLS7zurich', 'MSLS29berlin', 'MSLS30paris', 'MSLS23helsinki', 'MSLS24budapest', 'MSLS15london', 'MSLS34nairobi', 'MSLS21toronto', 'MSLS5manila', 'MSLS33goa', 'MSLS22ottawa', 'MSLS5nairobi', 'MSLS27goa', 'MSLS8budapest', 'MSLS28saopaulo', 'MSLS0zurich', 'MSLS29paris', 'MSLS30goa', 'MSLS26trondheim', 'MSLS16bangkok', 'MSLS25nairobi', 'MSLS9zurich', 'MSLS30zurich', 'MSLS18trondheim', 'MSLS32berlin', 'MSLS11helsinki', 'MSLS9boston', 'MSLS12tokyo', 'MSLS31bangkok', 'MSLS2manila', 'MSLS9moscow', 'MSLS23trondheim', 'MSLS6trondheim', 'MSLS30manila', 'MSLS21nairobi', 'MSLS34london', 'MSLS4goa', 'MSLS14budapest', 'MSLS26ottawa', 'MSLS22paris', 'MSLS0budapest', 'MSLS10melbourne', 'MSLS10manila', 'MSLS12melbourne', 'MSLS33helsinki', 'MSLS4phoenix', 'MSLS31tokyo', 'MSLS7helsinki', 'MSLS21austin', 'MSLS10phoenix', 'MSLS26budapest', 'MSLS17tokyo', 'MSLS18ottawa', 'MSLS35tokyo', 'MSLS13amsterdam', 'MSLS12phoenix', 'MSLS12ottawa', 'MSLS33melbourne', 'MSLS8london', 'MSLS12amman', 'MSLS5phoenix', 'MSLS26london', 'MSLS24manila', 'MSLS15tokyo', 'MSLS2bangkok', 'MSLS31budapest', 'MSLS12paris', 'MSLS5london', 'MSLS22manila', 'MSLS5berlin', 'MSLS14amman', 'MSLS2berlin', 'MSLS5amsterdam', 'MSLS17austin', 'MSLS25helsinki', 'MSLS34ottawa', 'MSLS15toronto', 'MSLS6moscow', 'MSLS33toronto', 'MSLS28bangkok', 'MSLS13saopaulo', 'MSLS11london', 'MSLS2london', 'MSLS24moscow', 'MSLS5paris', 'MSLS1budapest', 'MSLS12toronto', 'MSLS16paris', 'MSLS19paris', 'MSLS0tokyo', 'MSLS4melbourne', 'MSLS6austin', 'MSLS28moscow', 'MSLS9austin', 'MSLS3melbourne', 'MSLS18amman', 'MSLS25zurich', 'MSLS15ottawa', 'MSLS14boston', 'MSLS10paris', 'MSLS22berlin', 'MSLS31boston', 'MSLS1ottawa', 'MSLS26amman', 'MSLS25berlin', 'MSLS18zurich', 'MSLS13berlin', 'MSLS35bangkok', 'MSLS28phoenix', 'MSLS23manila', 'MSLS3paris', 'MSLS5toronto', 'MSLS24toronto', 'MSLS9melbourne', 'MSLS6goa', 'MSLS17amman', 'MSLS12budapest', 'MSLS13boston', 'MSLS0saopaulo', 'MSLS31phoenix', 'MSLS29amman', 'MSLS15helsinki', 'MSLS22amsterdam', 'MSLS27helsinki', 'MSLS29budapest', 'MSLS29austin', 'MSLS35toronto', 'MSLS35goa', 'MSLS32phoenix', 'MSLS32helsinki', 'MSLS34amsterdam', 'MSLS33paris', 'MSLS9ottawa', 'MSLS17amsterdam', 'MSLS14trondheim', 'MSLS21phoenix', 'MSLS13nairobi', 'MSLS20goa', 'MSLS21moscow', 'MSLS2amman', 'MSLS21zurich']


EXTRA_DATASETS_SF = ["SFXL0", "SFXL1", "SFXL2"]

EXTRA_DATASETS_PITTS = ['Pittsburgh2A','Pittsburgh2B','Pittsburgh2C','Pittsburgh2D',]

COMPARISON_FOR_CM = EXTRA_DATASETS_MSLS# + TRAIN_CITIES

EXTRA_DATASETS = EXTRA_DATASETS_PITTS + EXTRA_DATASETS_SF + EXTRA_DATASETS_MSLS

GPMS_DATASETS = EXTRA_DATASETS + TRAIN_CITIES

class GSVCitiesDataset(Dataset):
    def __init__(self, args, cities=['London', 'Boston'], img_per_place=4, min_img_per_place=4): # 4 for SuperPlace
        super(GSVCitiesDataset, self).__init__()
        self.base_path = os.path.join(args.datasets_folder, "gsv_cities/")
        self.cities = cities
        self.is_inference = False

        assert img_per_place <= min_img_per_place, f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.transform = transforms.Compose([transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),])

        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()

        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')

    def __getitem__(self, index):

        if self.is_inference:
            place_id = self.places_ids[index]
            row = self.dataframe.loc[place_id].iloc[0]
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)
            img = self.transform(img)

            return img, torch.tensor([place_id])


        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        place = place.sample(n=self.img_per_place)
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            img = self.transform(img)

            imgs.append(img)

        # [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        # row.name is the index of the row, not to be confused with image name
        pl_id = row.name % 10**5
        pl_id = str(pl_id).zfill(7)

        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
