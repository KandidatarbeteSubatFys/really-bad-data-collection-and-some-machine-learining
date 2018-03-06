
import sys
import os
import numpy as np
def main():
    #sys.argv is list of arguments. First argument (sys.argv[1]) is the maximum
    #number of guns used in any of the root files. Then you can add as many root
    #files (exempel.root) that you want to read. The last three arguments are the
    #names of the output files (exempel.txt). The first output is the XBe-data, the
    #second is the gundata and the last is the XBEsum-data. So for exampel a
    #possible input from the terminal is:
    #python3 gamma_auto_home.py 3 E1.root E2.root xb.txt gun.txt sum.txt

    #number_of_partices=number of maximum guns used in any file
    number_of_particles=sys.argv[1] 
    root_files=["" for x in range(len(sys.argv)-5)]
    output_files=["" for x in range(3)]
    for i in range((len(sys.argv)-5)):
        root_files[i]=sys.argv[i+2]
    #output files are made. The first is for the XBe data, the second for the gun
    #data and the third is for the XBEsum data
    for i in range(3):
        output_files[i]=sys.argv[(len(sys.argv)-3+i)]
        if os.path.isfile(output_files[i]):
            #clears previous output files
            open(output_files[i],'w').close() 
    
    for i in range((len(sys.argv)-5)):
        #converts one root file to txt
        make_txt(root_files[i]) 
        #adds right amount of zeros to gunfile. Comment this line and do one
        #modification to the make_txt function as well to not add zeros.
        add_zeros_and_shuffle_gun(number_of_particles)
        #appends the data from one root file to final output files
        add_to_final_files(output_files) 


#convers one root file to three txt files        
def make_txt(file):
    #h102 class is made from terminal
    os.system("root -q \'gamma_root_make_class.C(\""+file+"\")\'")
    #the code in the default h102.C class is replaced
    with open("gamma_h102_backup.C") as f:
        with open("h102.C", "w") as f1:
            for line in f:
                f1.write(line)
    #the Loop() method is run in root from terminal
    os.system("root -q gamma_root_loop.C")
    

#add zeros to gun data depending on the number of particles
def add_zeros_and_shuffle_gun(number_of_particles):
    with open("gamma_gunTVals.txt") as f:
        with open("tempfile.txt","w") as f1:
            for line in f:
                #line includes \n so removes it with rstrip()
                line=line.rstrip()
                tmp_string_list=line.split(' ')
                tmp_float_list=list(map(float, tmp_string_list))
                tmp_float_matrix=np.reshape(np.array(tmp_float_list),(int(len(tmp_string_list)/2),2))
                tmp_float_matrix_sorted=sorted(tmp_float_matrix,key=lambda x: x[0],reverse=True)
                tmp_float_list_sorted=np.reshape(tmp_float_matrix_sorted,(1,len(tmp_string_list)))
                sorted_line=" ".join(map(str,tmp_float_list_sorted[0]))
                f1.write(sorted_line+(2*int(number_of_particles)-len(tmp_string_list))*" 0"+" \n")
                #f1.write(line + (2 * int(number_of_particles) - len(tmp_string_list)) * " 0" + " \n")


#adds the final data from one root file to the final ouput files
def add_to_final_files(output_files):
    #appends the XBe data
    with open("gamma_XBe.txt") as f: 
        with open(output_files[0],"a") as f1:
            for line in f:
                f1.write(line)
    #appends the gun data
    with open("tempfile.txt") as f: #change to gamma_gunTVals if not adding zeros
        with open(output_files[1],"a") as f1:
            for line in f:
                f1.write(line)
    #appends the XBEsum data
    with open("gamma_sum_of_dep_energies.txt") as f:
        with open(output_files[2],"a") as f1:
            for line in f:
                f1.write(line)


if __name__=="__main__":
    main()

