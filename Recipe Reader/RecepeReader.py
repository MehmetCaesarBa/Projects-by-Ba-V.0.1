OrderRecipe = open("OrderRecipe.txt","w+")
#Order Recipe creates text document for each run, then it deletes the previous one
OrderRecipe.write("")
textVar = ["category","product","portion"]
TextName = ["categories.txt","products.txt","portions.txt"]
Stages = [0,1,2,3]
#Stages is the main column of this code. It determines instruction which will work in each step.
decoration1 = ("~"*15)
decoration2 = ("--"*30)
decoration3 = ("##"*124)
#Decorations are helpful to divide outputs.
total = float(0)
#Total variable sum all the prices and gives output

def prepareInfo(TextName,selection):
    file = open(TextName,"r+")  
    GonnaPrint = [] 
    for line in file:
        ElementsList = line.strip("# \n").split(";")
        if selection == "":
            GonnaPrint.append(ElementsList)
        else:
            for i in range(0,3):
                if selection == ElementsList[i]:
                    GonnaPrint.append(ElementsList)   
    return GonnaPrint          
#prepareInfo function assist to open given texts and give an output from user's selection

def getUserInput(Stage):
    if Stage == 0 or Stage == 1 or Stage == 2:
        question = input(f"What will you prefer as {textVar[Stage]} ?\n")
        return question
    elif Stage == 3:
        OkOrCont = input ("Would you like to continiue to select from menu? \n Yes or no? (y,n)\n").lower()
        return OkOrCont
#This function asks for input for each stage
    
def printMenu():
    global total
    selection = ""
    for Stage in range(0,3):
        var = prepareInfo(TextName[Stage],selection)
        for j in range(0,len(var)):
            print (f"{j+1}.{var[j][1]}")
        print(f"{decoration2}")
        answer = int(getUserInput(Stage))
        ChoosenOne = var[answer-1] 
        print(f"{decoration2}\n {ChoosenOne[1]}\n{decoration2}")
        OrderRecipe = open("OrderRecipe.txt","a+")
        OrderRecipe.write(f"{ChoosenOne[1]};")
        if Stage == 0:
            selection = ChoosenOne[0]
        elif Stage == 1:
            selection = ChoosenOne[2]
        elif Stage == 2:
            OrderRecipe.write(f"{ChoosenOne[2]}$ \n")
            total += float(ChoosenOne[2])
#This function prints values which are given from previous functions then it examines them to print a conclusion
    
def Main():
    print(f"{decoration1} Menu {decoration1} \n {decoration2}")
    Loop = True
    while Loop == True:
        
        MainVar = printMenu()
        print(MainVar)
        Ask = getUserInput(Stages[3])
        print(Ask)
        while type(Ask == "n" or Ask == "y") == False:
            Ask = getUserInput(Stages[3])
            print(Ask)
        if Ask == "y":
            Loop = True
        elif Ask == "n":
            Loop = False
    OrderRecipe = open("OrderRecipe.txt","r+")
    print(f"{decoration1}\nOrder Recipe \n {decoration3}")
    space = " "
    for line in OrderRecipe:
        FinalBill = line.strip().splitlines()
        
        for i in range(0,len(FinalBill)):
            PrintedBill = FinalBill[i].strip().split(";")
            for j in range(0,len(PrintedBill)):
                if j == int(len(PrintedBill)-1):
                    print(PrintedBill[j]+space*int(31-len(PrintedBill[j])))
                else:
                    print(PrintedBill[j]+space*int(31-len(PrintedBill[j])),end=" ")
        #These "for" loops assist to print order recipe in order.
                
    print(f"{decoration3}\n Total: {total}$")
#Main function run all functions under the same function and create loop for user to select multiple options. After loop, it gives order recipe as output.
Main()

    
        




