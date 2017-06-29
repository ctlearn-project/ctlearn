import os

for png in os.listdir('/home/gemini/energy_data/test/54/gamma-diffuse'):
    os.rename('/home/gemini/energy_data/test/54/gamma-diffuse/'+png,'/home/gemini/energy_data/training/54/gamma-diffuse/'+png)
for png1 in os.listdir('/home/gemini/energy_data/test/54/proton'):
        os.rename('/home/gemini/energy_data/test/54/proton/'+png1,'/home/gemini/energy_data/training/54/proton/'+png1)
for png2 in os.listdir('/home/gemini/energy_data/validation/54/gamma-diffuse'):
        os.rename('/home/gemini/energy_data/validation/54/gamma-diffuse/'+png2,'/home/gemini/energy_data/training/54/gamma-diffuse/'+png2)
for png3 in os.listdir('/home/gemini/energy_data/validation/54/proton'):
        os.rename('/home/gemini/energy_data/validation/54/proton/'+png3,'/home/gemini/energy_data/training/54/proton/'+png3)
