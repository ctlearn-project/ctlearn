/* something for working with ROOT files */
#include <TString.h>
#include <TChain.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>
#include <math.h>

using namespace std;

int meera_generateBinLists()
{
    TString mypath1 = "/data/nieto/mscw_energy_diffuse/gamma-diffuse*.root";
    string label1 = "gamma-diffuse";

    TString mypath2 = "/data/nieto/mscw_energy_diffuse/proton*.root";
    string label2 = "proton";

    TString st = "data";

    TChain ch1(st);
    ch1.Add(mypath1);
    TChain ch2(st);
    ch2.Add(mypath2);

    //energy bins
 //   float min_energy[3] = {0.1, 0.31,1};
 //   float max_energy[3] = {0.31, 1, 10};
    float min_energy[80];
    float max_energy[80];
    float j,k,l,m;
    for(int i=0; i<80; i++)
    {
        j = -2+i*0.05;
        k= pow(10,j);
        l=-2+(i+1)*0.05;
        m= pow(10,l);
        min_energy[i]=k;
        max_energy[i]=m;
    }
    for (int i = 0; i <80; i++)
    { string cut_selection_conditions = "ErecS>" + std:: to_string(min_energy[i]) + " && ErecS<" +std::to_string(max_energy[i])+ " && MSCW>-2.0 && MSCL>-2.0 && MSCL<5.0 && EChi2S>=0.0 && ErecS>0.0 &&EmissionHeight>0.0 && EmissionHeight<50.0 && sqrt(MCxoff^2 + MCyoff^2)<3.0 && sqrt(MCxoff^2 +MCyoff^2)>=0.0 && NImages>=3.0 &&dES>=0.0" ;
        
        string list = "_"+ std::to_string(i)+".txt";
        string filename1 = label1 + list;
        string filename2 = label2 +list;

        ((TTreePlayer*)(ch1.GetPlayer()))->SetScanRedirect(true);
        ((TTreePlayer*)(ch1.GetPlayer()))->SetScanFileName(filename1.c_str());
        ch1.Scan("runNumber:eventNumber",cut_selection_conditions.c_str());
        
        ((TTreePlayer*)(ch2.GetPlayer()))->SetScanRedirect(true);
        ((TTreePlayer*)(ch2.GetPlayer()))->SetScanFileName(filename2.c_str());
        ch2.Scan("runNumber:eventNumber",cut_selection_conditions.c_str());
    }

    return 0;
}
