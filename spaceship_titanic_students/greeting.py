import argparse

""" parser = argparse.ArgumentParser(description="Greet someone")
parser.add_argument("name", nargs="?", default="world", help="who to greet")


parser.add_argument("--times", type=int, default=1, help="how many times")



    
parser.add_argument("--shout", action="store_true", help="use capital letter")
args = parser.parse_args()

message = f"Hello {args.name}"
if args.shout:
    message = message.upper()
for _ in range(args.times):
     print(message)
    
    
print("--" * 40)
print(args)
     """
     
def cmd_hello(args):
    msg = f"Hello, {args.name}!"
    print(msg.upper() if args.shout else msg)
    
def cmd_add(args):
    print(args.x + args.y)

parser = argparse.ArgumentParser(description="Tiny Demo")
sub = parser.add_subparsers(dest="cmd", required=True) #sub is a router: an object that holds a lookup table of subcommand names to parsers.

p_hello = sub.add_parser("hello", help="say hello")   #creates a separate mini parser
p_hello.add_argument("name")
p_hello.add_argument("--shout", action="store_true", help="use capital letters")
p_hello.set_defaults(function=cmd_hello)  #attaches the function to run for that command

p_add = sub.add_parser("add", help="add two numbers")
p_add.add_argument("x", type=float)
p_add.add_argument("y", type=float)
p_add.set_defaults(function=cmd_add) #stores a function as an attribute on args:

print("Sub choices: " ,sub.choices)
print("--" * 40)
print("Parser: " ,parser)
args = parser.parse_args() #returns a namespace object exp: namespace(name='Doga', shout=True)
args.function(args)        #equivalent to -> func_name = args.function; func_name(args)