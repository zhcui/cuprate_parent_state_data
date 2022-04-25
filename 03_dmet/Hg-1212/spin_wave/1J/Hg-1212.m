AFsq = spinw;
AFsq.genlattice('lat_const',[3.8556000000 3.8556000000 3.1805000000],'angled',[90 90 90],'spgr',0)
AFsq.addatom('r',[0 0 0],'S', 0.5,'label','Cu1','color','b')
AFsq.table('atom')
plot(AFsq)
swplot.zoom(1.5)

AFsq.gencoupling('maxDistance',6.0)
AFsq.table('bond',[])

AFsq.addmatrix('label','J1','value',   122.059415, 'color','red')
AFsq.addcoupling('mat','J1','bond',2)
plot(AFsq,'range',[2 2 2])

AFsq.genmagstr('mode','direct','k', [1/2, 1/2, 1/2] ,'n',[0 0 1],'S',[1/2 -1/2 -1/2 1/2 -1/2 1/2 1/2 -1/2; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0],'nExt',0.01);
disp('Magnetic structure:')
AFsq.table('mag')

AFsq.energy
plot(AFsq,'range',[2 2 2])
Lz = 0.0
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
