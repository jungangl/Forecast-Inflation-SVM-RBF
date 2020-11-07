using BlsData

open(joinpath(homedir(), ".blsdatarc"), "w") do f
   write(f, "5ce1ac07f4264270b9a760d7dee08661") end


b = Bls()
result = get_data(b, "LNS11000000")
b = Bls(key="5ce1ac07f4264270b9a760d7dee08661")
