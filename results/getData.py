

# inputFile = open('resnet_n=7_90_39%.txt', 'r')
# outputFile = open('n=7_results.txt', 'w') 

# inputFile = open('n=7_initialData.txt', 'r')
# outputFile = open('n=7_initial_results.txt', 'w') 

# inputFile = open('resnet_n=9_88_86%.txt', 'r')
# outputFile = open('n=9_results.txt', 'w')

# inputFile = open('resnet_n=18_87_66%.txt', 'r')
# outputFile = open('n=18_results.txt', 'w')

# inputFile = open('resnet_n=5_90_64%.txt', 'r')
# outputFile = open('n=5_results.txt', 'w')

inputFile = open('resnet_n=3_89_58%.txt', 'r')
outputFile = open('n=3_results.txt', 'w')

lines = inputFile.readlines() 
numItString = "NumIterations:"
epochString = "Epoch"
imagesString = "images: "
percent = "%"
for line in lines:
    numItIndex = line.find(numItString)
    epochIndex = line.find(epochString)
    imageIndex = line.find(imagesString)
    percentIndex=  line.find(percent)
    if numItIndex != -1:
        iterations = line[numItIndex+len(numItString):epochIndex].strip()
        accuracy = line[imageIndex + len(imagesString):percentIndex].strip()
        resultString = iterations + "," + accuracy + "\n"
        print(resultString)
        outputFile.write(resultString)
