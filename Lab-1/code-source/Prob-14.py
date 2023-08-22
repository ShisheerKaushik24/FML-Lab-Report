
class Rect():
    def __init__(self,width,length,height):
        self.width=width
        self.length=length
        self.height=height

    def area1(self):
        return self.length*self.width
    def peri1(self):
        return 2*(self.length+self.width)
    def vol1(self):
        return self.length*self.width*self.height
    
length=int(input('Enter the intended length: '))
width=int(input('Enter the intended width: '))
height=int(input('Enter the intended height: '))
mensuration=Rect(width,length,height) 

print('The given value of length is: {}\nThe given value of Width is: {}\nThe given value of Height is: {}' .format(length,width,height))
rec_area=mensuration.area1()
print('The Area of Rectangle is: {}' .format(rec_area))
rec_peri=mensuration.peri1()
print('The Perimeter of Rectangle is: {}' .format(rec_peri))
rec_vol=mensuration.vol1()
print('The Volume of Rectangle is: {}' .format(rec_vol))