#Operator notations: "," Which divides different identifiers
import os

recognized_operands = ["==","!="]
CreatedTextsAsList = []
FileDictionary = {}
id_counter = 1

# I have defined assistant functions for making easier to understand code.
def ID_number(Identifiers,function):
    global id_counter
    if function == "add":
        FileDictionary.update({id_counter:Identifiers})
        ID_assigned = id_counter
        id_counter += 1
        return ID_assigned
    elif function == "remove":
        FileDictionary.pop(Identifiers,None)

def Table_Creator(row_of_table,column_of_table,number_of_lines):
    conditional_lines = len(row_of_table)
    print(f"Number of lines in file students: {number_of_lines}")
    print(f"Number of lines that hold the condition: {conditional_lines}")
    table_elements,length_of_elements = [],[]
    for line in row_of_table:
        disposable_line = line.strip(",").split(",")
        for number in column_of_table:
            table_elements.append(disposable_line[number])
            length_of_elements.append(len(disposable_line[number]))
        max_gap = max(length_of_elements)
    printer_list = []
    for line in row_of_table:
        printer_string = " "
        disposable_line = line.strip(",").split(",")
        if line == row_of_table[0]:
            
            for number in column_of_table:
                if number == column_of_table[0]:
                    gap_length = max_gap-len(disposable_line[number])
                    printer_string += "|"+f"{disposable_line[number]}"+(" "*gap_length)+"|"
                else:
                    gap_length = max_gap-len(disposable_line[number])
                    printer_string += f"{disposable_line[number]}"+(" "*gap_length)+"|"
            printer_list.append(len(printer_string)*"-")
            printer_list.append(printer_string)
            printer_list.append(len(printer_string)*"-")
        else:
            printer_string = " "
            disposable_line = line.strip(",").split(",")
            for number in column_of_table:
                if number == column_of_table[0]:
                    gap_length = max_gap-len(disposable_line[number])
                    printer_string += "|"+f"{disposable_line[number]}"+(" "*gap_length)+"|"
                else:
                    gap_length = max_gap-len(disposable_line[number])
                    printer_string += f"{disposable_line[number]}"+(" "*gap_length)+"|"
            printer_list.append(printer_string)
        
    for line in printer_list:
        print(line)

#It's important to warn user when it wants to write id as identifier
def Create(TextName,Identifiers):
    TextCreator = open(f"{TextName}.txt","w+")
    TextCreator.write("id,")
    
    for line in Identifiers:
        if line == Identifiers:
            print("You cannot create a file with attribute 'id'.")
            return
        else:
            TextCreator.write(f"{line},")
        
    TextCreator.close()
    if TextName in CreatedTextsAsList:
        print("There was already such a file. It is removed and then created again.")
    else:
        print("Corresponding file was successfully created.")
    CreatedTextsAsList.append(TextName)
    TextCreator.close()
   
def Delete(FileName):
    try :
        os.remove(f"{FileName}.txt")
        CreatedTextsAsList.remove(FileName)
    except FileNotFoundError:
        print("There is no such file.")
        return

#It'll display files and their identifiers.
def Display():
    file_names = os.listdir()
    i = 1
    for line in file_names:
        printedText = line.replace(".txt","")
        if printedText in CreatedTextsAsList :
            OpenText = open(line,"r+")
            OpenedTextIdentifiers = OpenText.read().splitlines()
            PrintedTextIdentifiers = OpenedTextIdentifiers[0]
            print(f"{i}){printedText}: {PrintedTextIdentifiers}")
            i += 1

def AddLine(Identifiers,TextName):
    if TextName in CreatedTextsAsList:
        OpenedText = open(f"{TextName}.txt","r+")
        Head_Identifiers = OpenedText.readlines()
        New_Head_Identifiers = Head_Identifiers[0].strip(",").split(",")
        EachIdentifier = Identifiers.strip(",").split(",")
        if len(New_Head_Identifiers) == (len(EachIdentifier) + 1) or len(New_Head_Identifiers) == (len(EachIdentifier) + 2) :
            AssignedID = ID_number(Identifiers,"add")
            OpenedText.write(f"\n {AssignedID},")
            for Identifier in EachIdentifier:
                OpenedText.write(f"{Identifier},")
            OpenedText.close()
            print(f"New line was successfully added to students with id = {AssignedID}.")
        else:
            print("Numbers of attributes do not match")
            return
    else:
        print("There is no such file.")
        return    
   
def RemoveLine(TextName,Operation):
    if TextName in CreatedTextsAsList:
        HeadIdentifier,Operand,Identifier = Operation.strip(" ").split(" ")
        text_reader = open(f"{TextName}.txt","r+").readline().strip(",").split(",")
        if HeadIdentifier in text_reader:
            removed_line_counter = 0
            line_number = text_reader.index(HeadIdentifier)
            text_line_reader = open(f"{TextName}.txt","r+").read().splitlines()
            for line in text_line_reader:
                
                if Operand == "==" and Identifier == line[line_number]:
                    text_line_reader.remove(line)
                    ID_number(line,"remove")
                    removed_line_counter += 1
                elif Operand == "!=" and Identifier != line[line_number]:
                    text_line_reader.remove(line)
                    ID_number(line,"remove")
                    removed_line_counter += 1
            
            text_destroyer = open(f"{TextName}.txt","w+")
            for line in text_line_reader:
                text_destroyer.write(f"{line}\n")
            print(f"{removed_line_counter} lines were successfully removed.")
            
        else:
            print("Your query contains an unknown attribute.")
            return
    else:
        print("There is no such file.")
        return
    
def Modify(TextName,requested_identifier,updated_identifier,Operation):
    if TextName in CreatedTextsAsList:
        HeadIdentifier,Operand,Identifier = Operation.strip(" ").split(" ")
        text_reader = open(f"{TextName}.txt","r+").readline().strip(",").split(",")
        if requested_identifier != "id":   
            if HeadIdentifier in text_reader:
                modified_line_counter = 0
                row_number = text_reader.index(HeadIdentifier)
                row_number_requested = text_reader.index(requested_identifier)
                text_line_reader = open(f"{TextName}.txt","r+").read().splitlines()
                reader_index=0
                for line in text_line_reader:
                    words_on_line = line.strip(" ").split(",")    
                    
                    if Operand == "==" and f"{Identifier}" == words_on_line[row_number]:
                        words_on_line[row_number_requested] = updated_identifier
                        modified_line = ",".join(words_on_line)
                        text_line_reader[reader_index] = modified_line
                        modified_line_counter += 1
                        
                    elif Operand == "!=" and f"{Identifier}" != words_on_line[row_number]:
                        words_on_line[row_number_requested] = updated_identifier
                        modified_line = " ".join(words_on_line)
                        text_line_reader[reader_index] = modified_line
                        modified_line_counter += 1
                    reader_index += 1

                
                text_destroyer = open(f"{TextName}.txt","w+")
                for line in text_line_reader:
                    text_destroyer.write(f"{line}\n")
                print(f"{modified_line_counter} lines were successfully  modified.")

            else:
                print("Your query contains an unknown attribute.")
                return


        else: 
            print("Id values cannot be changed.")
            return
                
    else:
        print("There is no such file.")
        return

def Fetch(TextName,Identifiers,Operation):
        if TextName in CreatedTextsAsList:
            HeadIdentifier,Operand,Identifier = Operation.strip(" ").split(" ")
            text_headline_reader = open(f"{TextName}.txt","r+").readline().strip(",").split(",")
            text_lines_reader = open(f"{TextName}.txt","r+").readlines() 
            number_of_lines = len(text_lines_reader)
            List_of_identifiers = Identifiers.strip(",").split(",")
            for Attributes in Identifiers:
                if Attributes not in text_headline_reader:
                    print("Your query contains an unknown attribute.")
                    return
            column_of_table,row_of_table = [],[text_lines_reader[0]]
            for element in text_headline_reader:
                if element == HeadIdentifier:
                    
                    row_number = text_headline_reader.index(HeadIdentifier)
                    if Operand == "==":
                        
                        for line in text_lines_reader[1:len(text_lines_reader)]:
                            disposable_line = line.strip(",").split(",")
                            if Identifier == disposable_line[row_number]:
                                row_of_table.append(line)
                    elif Operand == "!=":
                         for line in text_lines_reader[1:len(text_lines_reader)]:
                             disposable_line = line.strip(",").split(",")
                             if Identifier != disposable_line[row_number]:
                                row_of_table.append(line)
                    #This part creates a list which will use for creating table
                #Try elemnt in list like "na" which contains name
                
                if element in List_of_identifiers:
                    column_of_table.append(text_headline_reader.index(element))
                    
            Table_Creator(row_of_table,column_of_table,number_of_lines)
        else:
            print("There is no such file.")
            return

def Query_Function(Query):
    QueryList = Query.split(" ")
    if QueryList[0] == "create" :
        if QueryList[1]  == "file" and QueryList[3] == "with" and len(QueryList) == 5:
            TextName = QueryList[2]
            Identifiers = QueryList[4].strip(",").split(",")
            Create(TextName,Identifiers)
        else:
            print("Invalid query!")
            return
    elif QueryList[0] == "delete":
        if QueryList[1]  == "file" and len(QueryList) == 3:
             FileName = QueryList[2]
             Delete(FileName)
        else:
            print("Invalid query!")
            return
    elif QueryList[0] == "display":
        if QueryList[1]  == "files" and len(QueryList) == 2:
            Display()
        else:
            print("Invalid query!")
            return
    elif QueryList[0] == "add":
        if QueryList[2] == "into" and len(QueryList) == 4:
            Identifiers = QueryList[1] 
            TextName = QueryList[3]
            AddLine(Identifiers,TextName)
        else:
            print("Invalid query!")
            return
    elif QueryList[0] == "remove":
        if QueryList[1] == "lines" and QueryList[2] == "from" and QueryList[4] == "where" and QueryList[6] in recognized_operands and len(QueryList) == 8:
            TextName = QueryList[3]
            Operation = QueryList[5:9]
            RemoveLine(TextName,Operation)
        else:
            print("Invalid query!")
            return    
    elif QueryList[0] == "modify":
        if QueryList[2] == "in" and QueryList[4] == "as" and QueryList[6] == "where" and QueryList[8] in recognized_operands and len(QueryList) == 10:
            TextName = QueryList[3]
            requested_identifier = QueryList[1]
            updated_identifier = QueryList[5]
            Operation = QueryList[7:10]
            Modify(TextName,requested_identifier,updated_identifier,Operation)
        else:
            print("Invalid query!")
            return 
    
        #modify Identifier in Identifier as Identifier where Identifier Operator Identifier
    elif QueryList[0] == "fetch":
        if QueryList[2] == "from" and QueryList[4] == "where" and len(QueryList) == 8 and QueryList[6] in recognized_operands:
            TextName = QueryList[3] 
            Identifiers = QueryList[1]
            Operation = QueryList[5:8]
            Fetch(TextName,Identifiers,Operation)
        else:
            print("Invalid query!")
            return 
    else:
        print("Invalid query!")
        return

def Main():
    Looper = " "
    while Looper != "stop":
        Query = input("Please write your query:\n").lower()
        if Query == "x":
            break
        Query_Function(Query)
Main()
