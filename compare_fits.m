p1 = readmatrix('dat1_fit.txt');
p0 = readmatrix('dat0_fit.txt');

adjp = 1:12;
gamma = 13:19;
b1 = 20:26;
b2 = 27:33;
wout = 34:40;

p0([gamma,b1,wout]) = abs(p0([gamma,b1,wout]));
p1([gamma,b1,wout]) = abs(p1([gamma,b1,wout]));