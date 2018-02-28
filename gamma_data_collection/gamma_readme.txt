Skapar tre filer, en XBe fil, en gun fil och en XBe_sum file. Nollor läggs till i gunfilen. Klarar obegränsat med root-filer som
input. Ska någon simuleringarna bara ha en gun så skriv --tree=gunlist,FILE i ggland istället för bara filen för att skapa
en array istället för en float av gun-datan. Använder os.system() i python för att skriva i terminalen och detta har bara
testats i debian, men ska fungera för alla linux-system och windows.

Exempel: fyra root filer R1.root (4 guns) och R2.root (1 gun) R3.root (2 guns) R4.root (3 guns) vill skrivas till filer med namnen
output_XBe.txt, output_gun.txt och output_XBesum.txt (filnamnen är valbara, men antalet utdata filer måste vara tre och ordingen
är alltid XBe, gun och XBesum).Första argumentet är det maximala antalet guns som använts i någon av filerna. I detta exempel är det
ju 4. Så i rätt mapp skriver man:
python3 gamma_auto_home.py 4 R1.root R2.root R3.root R4.root output_XBe.txt output_gun.txt output_XBesum.txt 
