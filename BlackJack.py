import random
# Random has been imported because it needed for choosing random card
Player = str(input("Please write your name:\n"))
print("Welcome to 21 aka Blackjack")
quest1 = str(input("If you want to read info please press 1\nTo skip intro please press another button\n"))
#Information explains the game briefly
if quest1=="1":
    print("Blackjack is a card game, players try to acquire cards with a face value totalling 21 and no more.")
    print("If none of the players get the value of 21 player who has the nearest value to 21 will win.") 
    print("If the player exceed 21 other player will win.\nFor every round player choose a value of bet.\nBoth of the dealer and player gives money from their account to start the game. ")
Reset = "r"
Quit = ""
#Reset variable helps to create a loop for the player if it wants to continue the game. It resets the game and starts from the beginning
while Reset == "r" and Quit != "q":
    dealer = 1000
    player_credit = 1000
    #Round_ helps to demonstrate number of rounds
    Round_ = 0
    while dealer>0 and player_credit>0 and Quit != "q" :
        Round_ += 1 
        print("$#$#$#$$#$#$#$$#$#$#$$#$#$#$$#$#$#$$#$#$#$$#$#$#$$#$#$#$")
        print ("Round ",Round_)
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("Dealer's credit:",dealer)
        print("<*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*>")
        print(Player,"'s credit:",player_credit)
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        #Bet should be acceptable integer
        bet = int(input("Please write your bet to start(Maximum bet can be 1000 or greater than players or dealers credit): \n"))
        while bet > 1000 or bet > (player_credit or dealer) or bet <= 0:
            bet = int(input("Please write your bet to start(Maximum bet can be 1000 or greater than players or dealers credit): \n"))
        dealer -= bet
        player_credit -= bet
        
        #These limit variables will use for specific situation (If choosen cards were in order of Ace-Ace-10)
        Limit = 0
        LimitDealer = 0
        ace = 11
        king = 10
        quenn = 10
        jack = 10
        Cards = [ace,2,3,4,5,6,7,8,9,jack,quenn,king]
        #ChoosenCard is the variable which is choosen by computer randomly 
        
        Choosencard = random.choice(Cards)
        Dealer1 = Choosencard 
        Choosencard = random.choice(Cards)
        DealerChoosenCards = ""
        if Choosencard == ace:
            DealerChoosenCards += " - Ace "
        else:
            DealerChoosenCards += " - " + str(Choosencard)
        Double = ""
        ScoreD = Dealer1 + Choosencard
        if (Dealer1 == ace and Choosencard != ace) or (Dealer1 != ace and Choosencard == ace) :
            LimitDealer = 1
        if Dealer1 == ace and Choosencard == ace:
            LimitDealer = 2
            ScoreD -= 10
        print("Dealer's cards are: \n?",DealerChoosenCards," Total(Except first card):",ScoreD-Dealer1)
        #TotalP demonstrates total points of the player and ScoreD demonstrates total points of the dealer
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        TotalP = 0 
        PlayersChoosenCard = ""
        
        
        for i in range (1,3):
            Choosencard = random.choice(Cards)
            TotalP += Choosencard
            if i ==1:
                if Choosencard == ace:
                    PlayersChoosenCard += "Ace"
                else:   
                    PlayersChoosenCard += str(Choosencard)
            else:
                if Choosencard == ace:
                    PlayersChoosenCard += " - Ace"
                else:   
                    PlayersChoosenCard += " - " + str(Choosencard)
                    
            if Choosencard == ace: 
                Limit += 1
                if TotalP > 21:
                    TotalP = 12
            print("Your card,",Player,":",Choosencard)
            print ("Your cards are ",PlayersChoosenCard," Total:",TotalP)
        print("<*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*>")
        
        
        #After the first cut results are comparing by codes in below. If the situations given below are existed, game starts a new round while conditions are maintaining.
        if TotalP == 21 and ScoreD == 21:
            print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nDraw")
            player_credit += bet
            dealer += bet
            Quit = input("To quit please press q:\n").lower()
            #Quit helps player to finish game before the bankruptcy
            if Quit == "q":
                print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)

        elif TotalP == 21 and ScoreD != 21:
            print("Dealer's cards are: ",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\n!!!BLACKJACK!!!\nYou won the game")
            player_credit += (bet*2)
            Quit = input("To quit please press q:\n").lower()
            
            if Quit == "q":
                print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)

        elif TotalP != 21 and ScoreD == 21:
            print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\n!!!BLACKJACK!!!\nDealer won the game")
            dealer += (bet*2)
            Quit = input("To quit please press q:\n").lower()
            if Quit == "q":
                print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)
        else:
            Double = 0
            if player_credit >= bet:
                Double = input("If you want to double your bet and hit one another card please press d:\n").lower()
                if Double == "d" :
                        player_credit -= bet
                        dealer -= bet
                        Choosencard = random.choice(Cards)
                        if Choosencard == ace:
                            Limit += 1
                        HitStand = "s" 
                        TotalP += Choosencard
                        if TotalP > 21:
                            # While loop helps to calculate total point by subtracting 10 from each aces when 21 was exceeded
                            while Limit > 0:
                                Limit -= 1
                                TotalP -= 10
                            if Choosencard == ace:
                                PlayersChoosenCard += " - Ace"
                            else:   
                                PlayersChoosenCard += " - " + str(Choosencard)
                            
                            print("Your cards are ",PlayersChoosenCard," Total:",TotalP)
                            print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
                        
                        
                        else:
                            if Choosencard == ace:
                                PlayersChoosenCard += " - Ace"
                            else:   
                                PlayersChoosenCard += " - " + str(Choosencard)
                            print("Your cards are ",PlayersChoosenCard," Total:",TotalP)
                            print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")                             
             
             
            #If player doesn't double the bet these section asks to hit or stand a card
            if Double != "d":
                HitStand = input("To hit another card please press h,to stand press any other button:\n").lower()

               
                while HitStand == "h":
                    Choosencard = random.choice(Cards)
                    if Choosencard == ace:
                        Limit += 1
                        TotalP += Choosencard
                        if TotalP > 21:
                            while Limit > 0:
                                Limit -= 1
                                TotalP -= 10
                            if Choosencard == ace:
                                PlayersChoosenCard += " - Ace"
                            else:   
                                PlayersChoosenCard += " - " + str(Choosencard)
                            print("Your cards are ",PlayersChoosenCard," Total:",TotalP)
                            print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

                    else:
                        TotalP += Choosencard
                        if TotalP > 21:
                            while Limit > 0:
                                Limit -= 1
                                TotalP -= 10
                        if Choosencard == ace:
                            PlayersChoosenCard += " - Ace"
                        else:   
                            PlayersChoosenCard += " - " + str(Choosencard)
                        print("Your cards are ",PlayersChoosenCard," Total:",TotalP)
                        if TotalP <= 21:
                            HitStand = input("To hit another card please press h,to stand press any other button::\n").lower()
                        else:
                            break
                    print("<*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*><*-*>")
                    
            
            #These section allows dealer to choose card. Dealer chooses card for only one condition. If total point of the player is greater than dealer's point it chooses card automatically.
            if TotalP <= 21:
                for i in range(0,21):
                    if ScoreD == i:
                        while ScoreD < TotalP:     
                            Choosencard = random.choice(Cards)
                            ScoreD += Choosencard
                            if Choosencard == ace:
                                LimitDealer += 1
                            
                                if ScoreD > 21:
                                    while LimitDealer > 0:
                                        LimitDealer -= 1
                                        ScoreD -= 10
                                DealerChoosenCards += " - Ace"
                                print("Dealer's cards are: \n?",DealerChoosenCards," Total(Except first card):",ScoreD-Dealer1)
                                print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
                            else: 
                                if ScoreD > 21:
                                    while LimitDealer > 0:
                                        LimitDealer -= 1
                                        ScoreD -= 10
                                DealerChoosenCards += " - " + str(Choosencard)
                                print("Dealer's cards are: \n?",DealerChoosenCards," Total(Except first card):",ScoreD-Dealer1)
                                print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")  
            #   Total points are comparing by the below to determine winner.             
            if TotalP>21:
                print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nDealer won the game")
                if Double == "d":
                    dealer += (bet*4)
                else:
                    dealer += (bet*2)
                Quit = input("To quit please press q:\n").lower()
                if Quit == "q":
                    print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)        
            if ScoreD>21:
                print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nYou won the game")
            # "D" has been assigned for Double button. Double allows player to increase bet but it limited player to choose only one card.    
                if Double == "d":
                    player_credit += (bet*4)
                else:
                    player_credit += (bet*2)
                Quit = input("To quit please press q:\n").lower()
                if Quit == "q":
                    print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)
            elif ScoreD>TotalP and ScoreD <= 21:
                print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nDealer won the game")
                if Double == "d":
                    dealer += (bet*4)
                else:
                    dealer += (bet*2)
                Quit = input("To quit please press q:\n").lower()
                if Quit == "q":
                    print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)
            elif ScoreD<TotalP and TotalP <= 21:
                print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nYou won the game")
                if Double == "d":
                    player_credit += (bet*4)
                else:
                    player_credit += (bet*2)
                Quit = input("To quit please press q:\n").lower()
                if Quit == "q":
                    print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)
            elif ScoreD == TotalP:
                print("Dealer's cards are:",Dealer1,DealerChoosenCards,"\nTotal score of dealer is ",ScoreD,"\nDraw")
                if Double == "d":
                    dealer += (bet*2)
                    player_credit += (bet*2)
                else:
                    dealer += bet
                    player_credit += bet
                Quit = input("To quit please press q:\n").lower()
                if Quit == "q":
                    print("Thank you for playing Blackjack with me! \n Your total credit is" ,player_credit)
    Reset = input("If you want to reset bankroll please press r:\n").lower()

                            


                        




    

    

    












