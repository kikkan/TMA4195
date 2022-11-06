close all

mrstModule add ad-core mrst-gui 

jsonfile = fileread('diffusion.json');
jsonstruct = jsondecode(jsonfile);

paramobj = ReactionDiffusionInputParams(jsonstruct);

G = cartGrid([88, 88, 3]);
G = computeGeometry(G);

paramobj.G = G;

paramobj = paramobj.validateInputParams();

model = ReactionDiffusion(paramobj);


% setup schedule
total = 50;
n  = 50;
dt = total/n;
step = struct('val', dt*ones(n, 1), 'control', ones(n, 1));

control = []; % hot fix
control.none = [];
schedule = struct('control', control, 'step', step);

% setup initial state

nc = G.cells.num;
vols = G.cells.volumes;

initcase = 3;
switch initcase
  case 1
    cA      = zeros(nc, 1);
    cA(1)   = sum(vols);
    cB      = zeros(nc, 1);
    cB(end) = sum(vols);
    cC = zeros(nc, 1);
  case 2
    cA = ones(nc, 1);
    cB = ones(nc, 1);
    cC = zeros(nc, 1);
  case 3
    cA = zeros(nc, 1);
    cA(19382) = 5000; %should simulate a single pop in the middle of the grid (50x50)
    cB = zeros(nc, 1);
    cB(1:7744) = 25e-3;
    cC = zeros(nc, 1);
end

initstate.A.c = cA;
initstate.B.c = cB;
initstate.C.c = cC;

% run simulation

nls = NonLinearSolver();
nls.errorOnFailure = false;

[~, states, ~] = simulateScheduleAD(initstate, model, schedule, 'NonLinearSolver', nls);

%lets create a second pop in the middle of the time span

cA = states{end}.A.c;
cB = states{end}.B.c;
cC = states{end}.C.c;
cA(21362) = cA(21362) + 5000;

initstate.A.c = cA;
initstate.B.c = cB;
initstate.C.c = cC;

[~, states1, ~] = simulateScheduleAD(initstate, model, schedule, 'NonLinearSolver', nls);


%%

% Remove empty states (could have been created if solver did not converge)
ind = cellfun(@(state) ~isempty(state), states);
states = states(ind);

figure(1); figure(2); figure(3);

zA = max(states{40}.A.c);
zB = max(states{40}.B.c);
zC = max(states1{end}.C.c);

for istate = 1 : numel(states)

    state = states{istate};

    A = zeros(88,88);
    B = zeros(88,88);
    C = zeros(88,88);
    for i = 0 : 87
        for j = 1 : 88
            A(i+1,j) = state.A.c(15488 + i*88+j);
            B(i+1,j) = state.B.c(i*88+j);
            C(i+1,j) = state.C.c(i*88+j);
        end
    end


    set(0, 'currentfigure', 1);
    cla
    %plot3(1:2500, 1:2500, state.A.c);
    surf(1:88,1:88,A)
    zlim([0,zA])
    colorbar
    title('A concentration')
    
    set(0, 'currentfigure', 2);
    cla
    %plotCellData(model.G, state.B.c);
    surf(1:88,1:88,B)
    zlim([0,zB])
    colorbar
    title('B concentration')

    set(0, 'currentfigure', 3);
    cla
    %plotCellData(model.G, state.C.c);
    surf(1:88,1:88,C)
    zlim([0,zC])
    colorbar
    title('C concentration')

    drawnow
    pause(0.1);

end

ind = cellfun(@(state) ~isempty(state), states1);
states1 = states1(ind);

for istate = 1 : numel(states1)

    state = states1{istate};

    A = zeros(88,88);
    B = zeros(88,88);
    C = zeros(88,88);
    for i = 0 : 87
        for j = 1 : 88
            A(i+1,j) = state.A.c(15488 + i*88+j);
            B(i+1,j) = state.B.c(i*88+j);
            C(i+1,j) = state.C.c(i*88+j);
        end
    end


    set(0, 'currentfigure', 1);
    cla
    %plot3(1:2500, 1:2500, state.A.c);
    surf(1:88,1:88,A)
    zlim([0,zA])
    colorbar
    title('A concentration')
    
    set(0, 'currentfigure', 2);
    cla
    %plotCellData(model.G, state.B.c);
    surf(1:88,1:88,B)
    zlim([0,zB])
    colorbar
    title('B concentration')

    set(0, 'currentfigure', 3);
    cla
    %plotCellData(model.G, state.C.c);
    surf(1:88,1:88,C)
    zlim([0,zC])
    colorbar
    title('C concentration')

    drawnow
    pause(0.1);
    
end