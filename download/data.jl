using ZipFile
using HTTP 


using HTTP
url = "https://dl.dropboxusercontent.com/s/xjkaebigky34i21/data.zip"
HTTP.download(url, "data.zip")

using ZipFile

z = ZipFile.Reader("data.zip")  # open the zip file

for f in z.files  # loop over the files in the zip file
    if f.name == "data/" 
        isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")
    elseif f.name[1:5] != "data/"
        nothing
    else
        data = read(f)  # read the file data
        open(f.name, "w") do out  # write the file data to a new file
            write(out, data)
        end
    end
end

close(z)  # close the zip file