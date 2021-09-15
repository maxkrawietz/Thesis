from PIL import Image
dataset_names = ["Adult_Income", "Default_Of_Credit", "German_Credit", "Heart_Diseases", "Heart_Failure", "Ricci_vs_Stefano", "Student_Performance"]
definitions_names = ["statistical parity", "predictive parity", "negative predictive parity", "equal opportunity", "predictive equality", "overall accuracy equality"]
for defi in definitions_names:
    images = []
    for data in dataset_names:
        im = Image.open("plots/"+data+ "/" + data+"_"+defi+"_difference.png")
        images.append(im)
    size = images[0].size
    merged = Image.new("RGB",(3*size[0],2*size[1]),(255,255,255))
    merged.paste(images[0],(0,0))
    merged.paste(images[1],(size[0],0))
    merged.paste(images[2],(2*size[0],0))
    merged.paste(images[3],(0,size[1]))
    merged.paste(images[4],(size[0],size[1]))
    merged.paste(images[5],(2*size[0],size[1]))
    merged.save("plots/overall/"+defi+"_differences.png")
    