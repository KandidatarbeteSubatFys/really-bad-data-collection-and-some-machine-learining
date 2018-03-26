Skapar tre filer, en med kristallenergierna, en med gun data och en med totala deponerade energin. Nollor läggs till i gun-filen så att 
alla rader får sammadimension. gun-filen innehåller energin, cos(theta) och phi (från -pi till pi) för varje gun. Programmet klarar
obegränsat med root-filer som input. Ska någon simuleringarna bara ha en gun så skriv --tree=gunlist,FILE i ggland istället för bara filen
för att skapa en array istället för en float av gun-datan. Använder os.system() i python för att skriva i terminalen och detta har bara
testats i debian, men ska fungera för alla linux-system (och windows). Funkar för både crystal ball och dali2. För crystal ball, ange XB
som andra argument, för för dali2 ange dali2.

De program som måste vara installerade är python och root. Har endast testats för python 3.5.3 med standardpaket
installerade. Programmet är testat och fungerar för root av version 6.12/04, men när programmet testades för version
6.04/04 fungerade det ej. De filer som ingår i programmet är
get_txt_from_sim_data.py, root_loop.C, XB_h102_backup.C, dali2_h102_backup.C och make_class.C.

Exempel: fyra root filer R1.root (1 guns) och R2.root (2 gun) R3.root (3 guns) R4.root (4 guns) vill skrivas till filer
med namnen output_crystal.txt, output_gun.txt och output_tot_dep.txt (filnamnen är valbara, men antalet utdata filer måste
vara tre och ordingen är alltid kristall energier, gun data och tot_dep). Alla root filer har skapats i crystal ball. Första argumentet är det maximala
antalet guns som använts i 
någon av filerna. I detta exempel är det ju 4. Så i rätt mapp skriver man:
python3 get_txt_from_sim_data.py 4 XB R1.root R2.root R3.root R4.root output_crystals.txt output_gun.txt output_totaldep.txt

Man behöver inte ordna root-filerna efter antal gun.

