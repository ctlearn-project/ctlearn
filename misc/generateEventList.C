#include <TString.h>
#include <TChain.h>
#include <TTree.h>
#include <iostream>
#include <TTreePlayer.h>
#include <string>

using namespace std;

int generateEventList(TString mypath1, string label1, TString mypath2, string label2, string selection)
{

    TString st = "showerpars";

    TChain ch1(st);
    ch1.Add(mypath1);
    TChain ch2(st);
    ch2.Add(mypath2);

    /**

    string list = "list.txt";
    string filename1 = label1 + list;
    string filename2 = label2 + list;

    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch1.GetPlayer()))->SetScanFileName(filename1.c_str());
    ch1.Scan("eventNumber",selection.c_str());

    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanRedirect(true);
    ((TTreePlayer*)(ch2.GetPlayer()))->SetScanFileName(filename2.c_str());
    ch2.Scan("eventNumber",selection.c_str());

    */


    ch1->SetBranchAddress

    return 0;
}

data_tree->SetBranchAddress("MCe0", &energy);
data_tree->SetBranchAddress("MCxcore", &x_core);
data_tree->SetBranchAddress("MCycore", &y_core);
data_tree->SetBranchAddress("eventNumber", &event_num);
data_tree->SetBranchAddress("ntel", &ntel);
data_tree->SetBranchAddress("ntel_data", &ntel_data);
data_tree->SetBranchAddress("tel_data", &tel_data);
data_tree->SetBranchAddress("Trace", trace);
data_tree->SetBranchAddress("ntrig", &ntrig);
data_tree->SetBranchAddress("ltrig_list",ltrig_list);
data_tree->SetBranchAddress("MCprim", &prim);
data_tree->SetBranchAddress("MCxoff", &x_off);
data_tree->SetBranchAddress("MCyoff", &y_off);
data_tree->SetBranchAddress("MCze", &ze);
data_tree->SetBranchAddress("MCaz", &az);

for (int i = start_entry; i < stop_entry; i++)
{
    data_tree->GetEntry(i);

    if (debug)
    {
        std::cout << i << " " << ntel << " " << ntel_data << " " << tel_data[0] << " " << energy << " " << x_core << " " << y_core << " " << event_num << std::endl;
        std::cout << "pedrm =" << ped_rm << std::endl;
        std::cout << "MCprim =" << prim << std::endl;
        std::cout << "ntrig =" << ntrig << std::endl;
        std::cout << "ltrig_list" << std::endl;
        for (UInt_t j = 0; j < MAX_TEL;j++)
        {
            std::cout << ltrig_list[j] << std::endl;
        }
        std::cout << "tel map" << std::endl;
        for (std::map<int,int>::iterator it=tel_map.begin(); it!=tel_map.end(); ++it)
        {
            std::cout << it->first << " => " << it->second << std::endl;
        }
    }

}

int main(int argc, char **argv)
{
    //TString mypath1 = "/data/nieto/deeplearning/evn/gamma/20deg/0deg/root";
    TString mypath1 = "/data/deeplearning/data/root-evndisp/gamma-diffuse/*.root";
    string label1 = "gamma";

    //TString mypath2 = "/data/nieto/deeplearning/evn/gamma/20deg/180deg/*root";
    TString mypath2 = "/data/deeplearning/data/root-evndisp/proton/*.root";
    string label2 = "proton";

    string cut_selection_conditions = "MCe0>1.5 || length>0.5 || width>0.2"; 

    return generateEventList(mypath1,label1,mypath2,label2,cut_selection_conditions);
}


