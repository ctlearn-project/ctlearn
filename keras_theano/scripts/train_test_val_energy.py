import os
import random
#f=open('info_file.txt','w')
for eng_bin in os.listdir('/home/gemini/energy_data/training'):
#    os.makedirs('/home/gemini/energy_data/test/'+eng_bin+'/proton/')
#    os.makedirs('/home/gemini/energy_data/test/'+eng_bin+'/gamma-diffuse/')
#    os.makedirs('/home/gemini/energy_data/validation/'+eng_bin+'/proton/')
#    os.makedirs('/home/gemini/energy_data/validation/'+eng_bin+'/gamma-diffuse/')
    proton_list = os.listdir('/home/gemini/energy_data/training/'+eng_bin+'/proton')
    gamma_list = os.listdir('/home/gemini/energy_data/training/'+eng_bin+'/gamma-diffuse')
    p_10p = int(round(0.1*len(proton_list)))
    g_10p=int(round(0.1*len(gamma_list)))
    p_val_list=[]
    p_test_list=[]
    g_val_list=[]
    g_test_list=[]
    #moving proton test & val
    for x in range (p_10p):
        ran_file=random.choice(proton_list)
        if ran_file not in p_val_list and ran_file not in g_test_list:
            p_val_list.append(ran_file)
    proton_list=set(proton_list)
#    f.write(eng_bin+'\t|\t'+str(len(proton_list))+'\t|\t'+str(p_10p)+'\t|\t'+str(len(gamma_list))+'\t|\t'+str(g_10p)+'\n')
#    print(len(p_val_list))
#    print(p_val_list)
    for p_val_file in p_val_list:
        os.rename('/home/gemini/energy_data/training/'+eng_bin+'/proton/'+p_val_file,'/home/gemini/energy_data/validation/'+eng_bin+'/proton/'+p_val_file)
        if p_val_file in proton_list:
            proton_list.remove(p_val_file)
    proton_list=list(proton_list)
    for y in range (p_10p):
        ran_file2=random.choice(proton_list)
        if ran_file2 not in p_test_list and ran_file2 not in p_val_list:
            p_test_list.append(ran_file2)
    for p_test_file in p_test_list:
        os.rename('/home/gemini/energy_data/training/'+eng_bin+'/proton/'+p_test_file,'/home/gemini/energy_data/test/'+eng_bin+'/proton/'+p_test_file)
    #moving gamma test & val
    #gamma validation
    for z in range (g_10p):
        ran_file3 = random.choice(gamma_list)
        if ran_file3 not in g_val_list and ran_file3 not in g_test_list:
            g_val_list.append(ran_file3)
    gamma_list=set(gamma_list)
    print(g_val_list)
    for g_val_file in g_val_list:
        os.rename('/home/gemini/energy_data/training/'+eng_bin+'/gamma-diffuse/'+g_val_file,'/home/gemini/energy_data/validation/'+eng_bin+'/proton/'+g_val_file)
        if g_val_file in gamma_list:
            gamma_list.remove(g_val_file)
    gamma_list=list(gamma_list)
    #gamma test
    for a in range (g_10p):
        ran_file4 = random.choice(gamma_list)
        if ran_file4 not in g_test_list and ran_file not in g_val_list:
            g_test_list.append(ran_file4)
    for g_test_file in g_test_list:
        os.rename('/home/gemini/energy_data/training/'+eng_bin+'/gamma-diffuse/'+g_test_file,'/home/gemini/energy_data/test/'+eng_bin+'/gamma-diffuse/'+g_test_file)

