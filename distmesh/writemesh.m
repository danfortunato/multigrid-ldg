function writemesh(filename,p,t)
%WRITEMESH  Write a mesh from DistMesh to a file.

[p,t] = fixmesh(p,t);

N = size(p,2);
if size(t,2) ~= N+1
    error('The dimensions of p and t are incompatible.');
end

file = fopen(filename,'w');
if file == -1
    error('Unable to write to file %s.', filename);
end
 
% Write headers
fprintf(file, '%i %i\n\n', size(p,1), size(t,1));

% Write p
format = repmat('%f ',1,N);
format = strcat(format(1:end-1),'\n');
fprintf(file, format, p');
fprintf(file, '\n');

% Write t
format = repmat('%i ',1,N+1);
format = strcat(format(1:end-1),'\n');
fprintf(file, format, t');
fclose(file);

end