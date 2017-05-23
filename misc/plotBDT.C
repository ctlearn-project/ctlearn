#include <TTree.h>
#include <iostream>
#include <TFile.h>
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

using namespace std;

int main(int argc, char **argv)
{
    int label;
    float classifier_value;

    //open the file
    TFile *f = TFile::Open("/data/nieto/TMVA/BDT-V2-ID0-IM3-d20170517-0.5/complete_BDTroot/BDT_0.root");
    if (f == 0) {
        printf("Error: cannot open file\n");
        return 0;
    }

    // Create tyhe tree reader and its data containers
    TTreeReader myReader("TestTree", f);

    TTreeReaderValue<int> classID = TTreeReaderValue<int>(myReader,"classID");
    TTreeReaderValue<float> BDT_0 = TTreeReaderValue<float>(myReader,"BDT_0");

    ofstream gammafile ("BDT_gamma.txt");
    ofstream protonfile("BDT_proton.txt");

    while (myReader.Next()) 
    {
        label = *classID;
        classifier_value = *BDT_0;

        if(label==0)
        {
            gammafile << classifier_value;
            gammafile << "\n";
        }
        else if(label==1)
        {
            protonfile << classifier_value;
            protonfile << "\n";
        }
        else
        {
        }
    }


    gammafile.close();
    protonfile.close();

    return 0;
}
