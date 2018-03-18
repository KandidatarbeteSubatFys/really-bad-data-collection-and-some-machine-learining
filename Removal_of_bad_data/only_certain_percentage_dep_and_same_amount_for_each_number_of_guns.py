#Takes in deposited engergy for each crystal (XBe) and the the total deposited energy (XBsum (which is just th esum of XBe))
# and the gun data (format: energy_1 cos(theta_1) energy_2 cos(theta_2) ... , with zeros filling out if #guns<(maximum of particles used)
# for many events. From the input data it selects the events where the total deponated energy is more than a specified
# percentage of the sum of the gun energies and writes them into three new textfiles. The written data also has the
# same number of events for the different number of guns used.

#Everything is called XB for Crystal Ball, but there shouldn't be difficult at all (probably it is generelized already)
# generelizing this to work on other detectors

#IMPORTANT: the data has to be ordered from lowest amount of guns to highest amounts of guns


import sys


#function used for getting a certain coulum from a matrix (list format matrix, not numpy)
def column(matrix, i):
    return [row[i] for row in matrix]


#function used for getting multiple rows from a matrix (list format matrix, not numpy).
def get_rows_in_interval(matrix,start_row, end_row):
    out=[]
    if end_row==-1:
        end_row=len(matrix)-1
    for i in range(start_row,end_row+1):
        out.append(matrix[i])
    return out


def main(percentage,XBefile,gunfile,XBsumfile,XBefile_out,gunfile_out,XBsumfile_out):
    with open(XBefile, 'r') as f_XBe:
        with open(gunfile, 'r') as f_gun:
            with open(XBsumfile, 'r') as f_XBsum:
                with open(XBefile_out, 'w') as f_XBe_out:
                    with open(gunfile_out, 'w') as f_gun_out:
                        with open(XBsumfile_out, 'w') as f_XBsum_out:
                            XBe_lines = f_XBe.readlines()
                            gun_lines = f_gun.readlines()
                            XBsum_lines = f_XBsum.readlines()
                            row_index_and_number_of_guns=[]

                            #for each line (same number of rows for gun data and XBe data) the data is converted from
                            #string format into lists of floats. If the total deposited energy is more than the choosen
                            #percentage, then the row index and the number of guns used in that row is saved. (Would be
                            #more effective to use XBsum here instead of summing the XBe data, but I don't bother changing
                            #that because then I would have to double check that everything works one more time c: )
                            for i in range(len(XBe_lines)):
                                line = XBe_lines[i].rstrip()
                                tmp_string_list = line.split(' ')
                                XBe_list=list(map(float, tmp_string_list))
                                line = gun_lines[i].rstrip()
                                tmp_string_list = line.split(' ')
                                gun_list = list(map(float, tmp_string_list))

                                #removes the cos(theta) data from the gun data
                                gun_list_without_theta=[]
                                for j in range(int(len(gun_list)/2)):
                                    gun_list_without_theta.append(gun_list[2*j])

                                sum_of_gun_energies=sum(gun_list_without_theta)
                                sum_of_XBe_energies=sum(XBe_list)
                                if sum_of_XBe_energies/sum_of_gun_energies > float(percentage)/100:
                                    #wan't to save the row index and the number of gun used
                                    #if #guns<(maximum number of particles used), then I can find the number of particles
                                    #by locating where the first zero is.
                                    if gun_list_without_theta[-1]==0:
                                        row_index_and_number_of_guns.append(
                                            [i, gun_list_without_theta.index(0)])
                                    #Otherwise the #guns is just equal to the length of the list
                                    else:
                                        row_index_and_number_of_guns.append([i, len(gun_list_without_theta)])

                            #saves the minium and maximum of guns used of all of the selected events
                            minumum_number_of_guns=min(column(row_index_and_number_of_guns,1))
                            maximum_number_of_guns = max(column(row_index_and_number_of_guns,1))

                            #here I find the total number of elements (i.e. events) that should be used so that
                            #I get the same number of events for each #guns. To do that I find the smalles number of
                            #events for any of the guns.
                            total_number_of_elements_per_guns=1e50
                            for i in range(minumum_number_of_guns,maximum_number_of_guns+1):
                                tmp=column(row_index_and_number_of_guns,1).count(i)
                                if tmp<total_number_of_elements_per_guns:
                                    total_number_of_elements_per_guns=tmp

                            #for each number of guns, total_number_of_elements_per_guns of events are saved.
                            row_index_for_even_amount_per_number_of_guns=[]
                            for i in range(minumum_number_of_guns, maximum_number_of_guns+1):
                                start_index = column(row_index_and_number_of_guns, 1).index(i)
                                end_index=start_index+total_number_of_elements_per_guns-1
                                row_index_for_even_amount_per_number_of_guns = row_index_for_even_amount_per_number_of_guns + get_rows_in_interval(
                                    row_index_and_number_of_guns, start_index, end_index)

                            #Then the selected data is printed into textfiles
                            for i in row_index_for_even_amount_per_number_of_guns:
                                f_XBe_out.write(XBe_lines[i[0]])
                                f_gun_out.write(gun_lines[i[0]])
                                f_XBsum_out.write(XBsum_lines[i[0]])






if __name__ == '__main__':
    percentage=sys.argv[1]
    XBefile=sys.argv[2]
    gunfile = sys.argv[3]
    XBsumfile = sys.argv[4]
    XBefile_out = sys.argv[5]
    gunfile_out = sys.argv[6]
    XBsumfile_out = sys.argv[7]
    main(percentage,XBefile,gunfile,XBsumfile,XBefile_out,gunfile_out,XBsumfile_out)

