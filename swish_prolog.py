% Facts: Train information
train(12951, 'Mumbai Central - Ahmedabad Duronto Express', 'Mumbai Central', 'Ahmedabad', 6.15, 12.35).
train(12901, 'Mumbai Central - New Delhi Rajdhani', 'Mumbai Central', 'New Delhi', 16.35, 7.25).
train(19023, 'Mumbai Central - Firozpur Janata Express', 'Mumbai Central', 'Firozpur', 22.45, 22.30).
train(12925, 'Mumbai Central - Jaipur Superfast', 'Mumbai Central', 'Jaipur', 20.50, 12.15).
train(12933, 'Mumbai Central - Amritsar Express', 'Mumbai Central', 'Amritsar', 15.45, 17.20).

% Facts: Station codes
station('Mumbai Central', 'BCT').
station('Ahmedabad', 'ADI').
station('New Delhi', 'NDLS').
station('Firozpur', 'FZR').
station('Jaipur', 'JP').
station('Amritsar', 'ASR').

% Rules for train queries
train_to(TrainNo, Name, Destination) :-
    train(TrainNo, Name, _, Destination, _, _).

train_from(TrainNo, Name, Source) :-
    train(TrainNo, Name, Source, _, _, _).

train_timing(TrainNo, Departure, Arrival) :-
    train(TrainNo, _, _, _, Departure, Arrival).

journey_time(TrainNo, Hours) :-
    train(TrainNo, _, _, _, Departure, Arrival),
    Hours is round((Arrival - Departure + 24) mod 24).

available_trains(Source, Destination, TrainNo, Name, Departure, Arrival) :-
    train(TrainNo, Name, Source, Destination, Departure, Arrival).

% Simple queries to test
find_train(Source, Destination) :-
    available_trains(Source, Destination, TrainNo, Name, Depart, Arrive),
    format('Train ~w: ~w (~2f - ~2f)~n', [TrainNo, Name, Depart, Arrive]).


?- train_to(12951, Name, Destination).
?- train_from(12901, Name, Source).
?- train_timing(19023, Departure, Arrival).
?- journey_time(12925, Hours).
?- available_trains('Mumbai Central', 'Ahmedabad', TrainNo, Name, Dep, Arr).
?- available_trains('Mumbai Central', 'New Delhi', TrainNo, Name, Dep, Arr).
?- station('Mumbai Central', Code).
?- station('Ahmedabad', Code).
?- available_trains('Mumbai Central', 'Ahmedabad', TrainNo, Name, Dep, Arr).
?- find_train('Mumbai Central', 'Ahmedabad').
?- find_train('Mumbai Central', 'New Delhi').






% Family Tree Facts
male('Krish').
male('Deepak').
male('Arvind').

female('Prijal').

parent('Deepak', 'Krish').    % Deepak is parent of Krish
parent('Prijal', 'Krish').    % Prijal is parent of Krish
parent('Arvind', 'Deepak').   % Arvind is parent of Deepak

% Family Relationship Rules
father(Father, Child) :-
    male(Father),
    parent(Father, Child).

mother(Mother, Child) :-
    female(Mother),
    parent(Mother, Child).

son(Son, Parent) :-
    male(Son),
    parent(Parent, Son).

daughter(Daughter, Parent) :-
    female(Daughter),
    parent(Parent, Daughter).

grandfather(Grandfather, Grandchild) :-
    male(Grandfather),
    parent(Grandfather, Parent),
    parent(Parent, Grandchild).

grandmother(Grandmother, Grandchild) :-
    female(Grandmother),
    parent(Grandmother, Parent),
    parent(Parent, Grandchild).

grandson(Grandson, Grandparent) :-
    male(Grandson),
    parent(Parent, Grandson),
    parent(Grandparent, Parent).

granddaughter(Granddaughter, Grandparent) :-
    female(Granddaughter),
    parent(Parent, Granddaughter),
    parent(Grandparent, Parent).

sibling(Sibling1, Sibling2) :-
    parent(Parent, Sibling1),
    parent(Parent, Sibling2),
    Sibling1 \= Sibling2.

% Query Examples to Test:
father(X, 'Krish').        
mother(X, 'Krish').       
grandfather(X, 'Krish').  
son(X, 'Deepak').          
grandson(X, 'Arvind').     






% Temperature conversion between Celsius, Fahrenheit, and Kelvin

% Facts
temperature_scale(celsius).
temperature_scale(fahrenheit).
temperature_scale(kelvin).

% Rules for conversion
convert(Temp, From, To, Result) :-
    From = To,
    Result = Temp.

convert(Temp, celsius, fahrenheit, Result) :-
    Result is (Temp * 9/5) + 32.

convert(Temp, fahrenheit, celsius, Result) :-
    Result is (Temp - 32) * 5/9.

convert(Temp, celsius, kelvin, Result) :-
    Result is Temp + 273.15.

convert(Temp, kelvin, celsius, Result) :-
    Result is Temp - 273.15.

convert(Temp, fahrenheit, kelvin, Result) :-
    convert(Temp, fahrenheit, celsius, C),
    convert(C, celsius, kelvin, Result).

convert(Temp, kelvin, fahrenheit, Result) :-
    convert(Temp, kelvin, celsius, C),
    convert(C, celsius, fahrenheit, Result).

% Helper predicate for easy conversion
temperature(Temp, From, To, Result) :-
    convert(Temp, From, To, Result),
    format('~w° ~w = ~2f° ~w~n', [Temp, From, Result, To]).




convert(100, celsius, fahrenheit, F).
convert(32, fahrenheit, celsius, C).
temperature(0, celsius, fahrenheit, Result).