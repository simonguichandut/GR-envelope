import sys
from env_GR import MakeEnvelope,pressure
from IO import load_params
M, RNS, y_inner, comp, save, img = load_params()

def driver_envelope(Rphotkm,p=0):

    from IO import write_to_file

    if Rphotkm == 'all': # a preset list of photospheric radii
        Rphotkm = (13,15,20,30,40,50,70,100,150,200)

    problems,success = [],[]

    for R in Rphotkm:
        print('\n*** Calculating envelope with photosphere at %d km***\n'%R)
        try:
            r,rho,T,Linf = MakeEnvelope(R,p=p)
            success.append(R)
            write_to_file(R,[r, rho, T, pressure(rho, T), Linf])
        except:
            problems.append(R)
            print('PROBLEM WITH Rphot = %d km'%R)
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found solutions for these values : ',success)
    if len(problems)>=1:
        print('There were problems for these values : ',problems,'\n')





# Command line call
if len(sys.argv)>1:
            
    i = sys.argv[1]
    
    if len(i)>3 and ',' not in i:
        sys.exit('Give Rphots separated by commas and no spaces')

    if i=='all':
        Rphotkm='all'
    else:
        if ',' in i:
            Rphotkm = [eval(x) for x in i.split(',')]
        else:
            Rphotkm = [eval(i)]

    if len(sys.argv)<3:
        driver_envelope(Rphotkm)
    else:
        if sys.argv[2]=='0' or sys.argv[2]=='1' or sys.argv[2]=='2':
            p = eval(sys.argv[2])
            driver_envelope(Rphotkm, p=p)
        else:
            sys.exit('To plot solution, give 1 (density) or 2 (Temperature')
        
        
        
        
        
        

