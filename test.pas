program test;
{$mode objfpc}{$H+}
uses Classes, SysUtils, fpctinn;

procedure main;
var
  t: TStrings;
begin
  writeln('Loading data');
  t := TStringList.Create;
  t.LoadFromFile('semeion.data');
  writeln('Done (',t.Count,' lines)');
  t.Free;
end;  

begin
  main;
end.
