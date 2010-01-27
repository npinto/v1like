
"""
Modules to help parse options for classes.
"""

from optparse import Option, OptionValueError, OptionParser
from copy import copy
import types, logging, logging.config, sys
try:
    import sinc.base.sinclib as sinclib
except:
    pass

try:
    logging.config.fileConfig('logger.conf')
except:
    pass

log = logging.getLogger('OptParserExtended')

def check_list(option, dummy, value):
    """
    Evaluate `value` as a list.
    """
    pre_list = value.split(',')
    ltype = option.ltype
    final_list = []
    if ltype != None:
        for i in pre_list:
            try:
                if ltype == "int":
                    final_list.append( int(float(i)) )
                elif ltype == 'float':
                    final_list.append( float(i) )
                else:
                    final_list.append( str(i) )
            except Exception, e:
                raise e
                #raise OptionValueError(
                #    "option %s: invalid type given, %s in value %s"
                #    % (opt, str(i), value) )
    else:
        final_list = pre_list
    return final_list

def check_tuple( option, opt, value ):
    """
    Evaluate `value` as a tuple.
    """
    my_tuple = tuple( check_list( option, opt, value ) )
    return my_tuple

def check_expr( dummy, opt, value ):
    """
    Evaluate `value` as an expression.
    """
    try:
        val = eval(value)
    except:
        raise OptionValueError( "Could not evaluate value for option %s:\n%s"
            % ( opt, value ) )
    return val

def check_boolean( dummy, opt, value ):
    """
    Cast `value` to a boolean value.
    """
    try:
        val = bool(value)
    except:
        raise OptionValueError("Could not evalute %s into boolean for " \
                               "option %s.\n" % (value, opt))
    return val

def check_function( dummy, opt, value ):
    """
    Cast `value` to a function.  Currently does nothing.
    """
    try:
        pass
    except:
        raise OptionValueError("Could not evaluate %s into function for " \
            "option %s.\n" % (value, opt ))
    return value

def check_eval(dummy, opt, value):
    """
    Takes `value` and evaluates it as a python expression.
    """
    try:
        val = eval(str(value))
    except:
        raise OptionValueError("Could not evaluate %s as a python " \
            "expression for option %s.\n" % (value, opt))
    return val

class OptionExtended( Option ):

    """
    Extend the Options class with a few new types.
    """

    def __init__( self, *args, **kwargs ):
        try:
            ltype = kwargs.pop('ltype')
        except:
            ltype = None
        Option.__init__( self, *args, **kwargs )
        self.ltype = ltype

    TYPES = Option.TYPES + ("tuple", "list", "expr", "bool", "function",
        "func", "eval") 
    TYPE_CHECKER = copy(Option.TYPE_CHECKER)
    TYPE_CHECKER["tuple"] = check_tuple
    TYPE_CHECKER["list"]  = check_list
    TYPE_CHECKER["expr"]  = check_expr
    TYPE_CHECKER["bool"]  = check_boolean
    TYPE_CHECKER["function"] = check_function
    TYPE_CHECKER["func"] = check_function
    TYPE_CHECKER["eval"] = check_eval

def test_options( ):
    
    """
    Simple test class.
    """
    
    option_list =     [
        OptionExtended("-M", "--M", action="store", dest="M", \
                       default=(1, 2, 3, 4), type="tuple",ltype="int" )
            ]
    parser = OptionParser(option_class = OptionExtended, 
        option_list = option_list)
    option, dummy = parser.parse_args(args=["-M", "2.0"])
    log.info('From Test Options class, M=%f, should equal 2.0', option.M)
    return option.M

def assign_opts(*args, **kwargs):
    """
    Assigns options.
    """
    if 'my_args' in kwargs.keys() and kwargs['my_args'] != None:
        my_args = kwargs.pop('my_args')
    else:
        my_args = sys.argv[1:] 
    option_defaults = {}
    if len( args ) == 1:
        option_defaults = args[0]
    else:
        option_defaults = args
    option_properties = {'functions':[], 'eval':[]}
    option_list = []
    copy_props = []
    for i in option_defaults.keys():
        copy_props.append( i )
        if len( option_defaults[i] ) < 4:
            raise ValueError("Not enough options passed in defaults for " \
                             "%s." % i)
        if option_defaults[i][0] != None:
            short = '-' + option_defaults[i][0]
        else:
            short = None
        long_opt = '--' + option_defaults[i][1]
        dest = str(i)
        desc = option_defaults[i][2]
        default = option_defaults[i][3]
        if len( option_defaults[i] ) > 4:
            default_type = option_defaults[i][4]
        else:
            default_type = None
        ltype = None
        if (default_type == "func" or default_type == "function"):
            option_properties['functions'].append( i )
        if default_type == "eval":
            option_properties['eval'].append( i )
        if (default_type == "list" or default_type == "tuple") \
                and len(option_defaults[i]) > 5:
            ltype = option_defaults[i][5]
        if default_type == None or default_type == types.NoneType:
            if short != None:
                opt = OptionExtended(short, long_opt, dest=dest, help=desc, \
                                    default=default)
            else:
                opt = OptionExtended(long_opt, dest=dest, help=desc, \
                                     default=default)
        elif ltype == None:
            if short != None:
                opt = OptionExtended(short, long_opt, dest=dest, help=desc, \
                                     default=default, type = default_type)
            else:
                opt = OptionExtended(long_opt, dest=dest, help=desc, \
                                     default=default, type = default_type)
        else:
            if short != None:
                opt = OptionExtended(short, long_opt, dest=dest, help=desc,
                                     default=default, type = default_type, 
                                     ltype = ltype) 
            else:
                opt = OptionExtended(long_opt, dest=dest, help=desc, 
                                     default=default, type = default_type,
                                     ltype = ltype)
        option_list.append( opt )
    parser = OptionParser(option_class = OptionExtended, \
                          option_list = option_list)
    options, args = parser.parse_args( args=my_args )

    if not hasattr(options, 'problem_file'):
        options.problem_file = None

    if ( options.problem_file != None ):
        options.problem_file = options.problem_file.replace('/','.')
        if options.problem_file.endswith('.py'):
            options.problem_file = options.problem_file[:-3]
        try:
            mod_list = options.problem_file.split('.')
            if len(mod_list) == 1 and mod_list[0].find('/') >= 0:
                mod_list = options.problem_file.split('/')
            #root = mod_list[0]
            prob_mod = __import__( options.problem_file )
            for mod_name in mod_list[1:]:
                prob_mod = getattr( prob_mod, mod_name )
        except Exception:
            logging.error("Couldn't find problem file: %s" % \
                          options.problem_file)
            raise
        for attr in option_defaults.keys():
            try:
                file_val = getattr(prob_mod, attr)
                parser.set_default(attr, file_val)
            except:
                pass
        options, args = parser.parse_args( args = my_args )
        for func_name in option_properties['functions']:
            try:
                attr_val = getattr( prob_mod, func_name )
                #logging.debug("Setting function: %s, value: %s", 
                #    func_name, attr_val )
                setattr( options, func_name, attr_val )
            except Exception, e:
                logging.debug("Function %s not found in module.", func_name)
                logging.debug(e)
                
    dims = getattr( options, 'dimensions', 1 )
    options.dimensions = dims


    # Make the correct dimensions:
    if dims != 1:
        for opt in parser.option_list:
            if opt.dest == None:
                continue
            val = getattr( options, opt.dest, None )
            if opt.type == "list" or opt.type == "tuple":
                if not (types.ListType <= type(val)):
                    val = ( val,  )
                if len( val ) == 1:
                    val *= dims
                elif len( val ) == dims:
                    pass
                else:
                    raise OptionValueError("Incorrect number of parameters"
                        " passed for %s" % opt.get_opt_string()    )
                setattr(options, opt.dest, val)


    # Make functions into actual python functions 
    i = None
    for i in option_properties['functions']:
        cur_val = getattr( options, i )
        if isinstance( cur_val, types.FunctionType ) or \
                isinstance( cur_val, types.MethodType ):
            continue
        try:
            setattr( options, i, globals[getattr(options, i)] )
        except:
            try:
                # Different calling syntaxes for 1-dim and n-dim functions
                if dims == 1:
                    cmd = 'def f(x):\n \t return ' + getattr(options, i)
                else:
                    cmd = 'def f(*args):\n'
                    # pylint: disable-msg=W0631
                    for j in range(dims):
                        cmd += '\t' + 'x' + str(j) + ' = args[' + str(j) + ']'
                    cmd += '\t' + 'return ' + getattr(options, j) 
                exec(cmd) #pylint: disable-msg=W0122
                setattr( options, i, f ) #pylint: disable-msg=E0602
            except:
                try:
                    setattr(options, i, globals[option_defaults[i]])
                except:
                    setattr(options, i, None)
        # Last chance to set the option
        try:
            getattr(options, i)
        except:
            setattr(options, i, None)
    
    # Evaluate expressions into floats.  Really only works with 1-dim
    if dims == 1 and option_properties.has_key('eval'):
        #pylint: disable-msg=W0631
        for opt in parser.option_list:
            if i in option_properties['eval'] and getattr(options, i) != None:
                try:
                    setattr(options, i, float(getattr(options, i)))
                except:
                    try:
                        setattr(options, i, eval(str(getattr(options, i))))
                    except Exception, e:
                        print "Unable to set %s to %s" % \
                            (str(i), str(getattr(options, i)))
                        raise e

    options.copy_props = copy_props

    # Assign the transforms:
    if getattr( options, 'transform', None ):
        options = assign_transforms( options )
    return options

def assign_transforms( options ):
    """
    Assign options using the transforms object.
    """
    mytransforms = ['phipp', 'phip', 'phi', 'f1', 'f2', 'f3', 'psi']
    options.copy_props += mytransforms
    if options.dimensions == 1 and \
            (not isinstance(options.transform, types.TupleType)):
        tf = sinclib.transforms( options.transform )
        for func_name in mytransforms:
            func = getattr( tf, func_name, None )
            if func == None:
                log.warning("Function %s not found for transform %s", \
                            (func_name, options.transform))
            setattr( options, func_name, func )
    else:
        count = 0
        for func_name in mytransforms:
            default = [ None for _ in range( options.dimensions ) ] 
            setattr( options, func_name, default )
        for dummy in options.transform:
            tf = sinclib.transforms( options.transform[count] )
            for func_name in mytransforms:
                func = getattr( tf, func_name, None )
                if func == None:
                    log.warning("Function %s not found for transform %s", \
                                (func_name, options.transform))
                mylist = getattr( options, func_name )
                mylist[count] = func 
            count += 1
    return options

class OptParserExtended( object ):
    
    """
    The OptParserExtended class assigns options based upon defaults that
    the user specifies.
    """

    @staticmethod
    def defaults():
        """
        The defaults method.  Subclasses should implement this.
        """

    def __init__(self, my_args=None, options=None):
        if options == None:
            options = self.assign_opts(self.defaults(), my_args=my_args)
            copy_props = options.copy_props
        else:
            try:
                copy_props = options.keys()
            except:
                copy_props = options.copy_props
        is_dict = type(options) == types.DictType
        class OptTemp: # pylint: disable-msg=W0232
            """
            Temporary class for assigning options.
            """
        self.options = OptTemp()
        self.options.copy_props = copy_props
        for prop in copy_props:
            try:
                if is_dict:
                    setattr( self, prop, options[prop] )
                    setattr( self.options, prop, options[prop] )
                else:
                    attr = getattr( options, prop )
                    setattr( self, prop, attr )
                    setattr( self.options, prop, attr )
            #except AttributeError, ae:
            #    logging.warning("Attribute %s not present in options!", prop )
            #    pass
            #except KeyError, ke:
            #    pass
            except (AttributeError, KeyError):
                pass

    def assign_opts( self, *args, **kw ):
        """
        Assigns options based on the default properties.
        """
        return assign_opts( *args, **kw )

