AFsq = spinw;
AFsq.genlattice('lat_const',[3.8556000000 3.8556000000 3.1805000000],'angled',[90 90 90],'spgr',0)
AFsq.addatom('r',[0 0 0],'S', 0.5,'label','Cu1','color','b')
AFsq.table('atom')
plot(AFsq)
swplot.zoom(1.5)

AFsq.gencoupling('maxDistance',8.0)
AFsq.table('bond',[])

Zc = 1.219
J1 = 155.388654 * Zc
J2 = 9.721381 * Zc
J3 = 9.721381 * Zc
Jc = 194.427622 * Zc
Jperp = 8.860774993064345 * Zc

AFsq.addmatrix('label','J1','value', (J1 - Jc * 0.5), 'color','red')
AFsq.addmatrix('label','J2','value', (J2 - Jc * 0.25), 'color','blue')
AFsq.addmatrix('label','J3','value', (J3), 'color','black')
AFsq.addmatrix('label','Jp','value', Jperp,'color','green')

AFsq.addcoupling('mat','J1','bond',2)
AFsq.addcoupling('mat','J2','bond',4)
AFsq.addcoupling('mat','J3','bond',8)
AFsq.addcoupling('mat','Jp','bond',1)
plot(AFsq,'range',[2 2 2])

AFsq.genmagstr('mode','direct','k', [1/2, 1/2, 1/2] ,'n',[0 0 1],'S',[1/2 -1/2 -1/2 1/2 -1/2 1/2 1/2 -1/2; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0],'nExt',0.01);
disp('Magnetic structure:')
AFsq.table('mag')

AFsq.energy
plot(AFsq,'range',[2 2 2])
Lz = 0.46
Qcorner = {[1/4 1/4 Lz] [1/2 0 Lz] [0 0 Lz] [1/4 1/4 Lz] 2001};
sqSpec = AFsq.spinwave(Qcorner, 'hermit', true);

Qlab  = {'R' 'X' '\Gamma' 'R'};
sqSpec = sw_neutron(sqSpec);
sqSpec = sw_egrid(sqSpec, 'Evect',linspace(0, 350, 1000));

omega = sqSpec.omega
hkl = sqSpec.hkl
swconv = sqSpec.swConv
swint = sqSpec.swInt
evect = sqSpec.Evect
sperp = sqSpec.Sperp
save("omega.mat", "omega")
save("hkl.mat", "hkl")
save("swconv.mat", "swconv")
save("sperp.mat", "sperp")
save("swint.mat", "swint")
save("evect.mat", "evect")

figure
[fhand, phand] = sw_plotspec(sqSpec, 'mode', 3, 'dashed',true,  'dE', 35, 'qlabel',Qlab)
