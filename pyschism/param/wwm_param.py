import os
import f90nml
from typing import Union
from datetime import timedelta, datetime
import json
import xarray as xr
import numpy as np

class WWM_Param:
    """
    Class for setting up WWM III namelist input file: wwminput.nml
    """

    def __init__(
        self,
        template: Union[f90nml.namelist.Namelist, str, os.PathLike],
        start_datetime: Union[int, float, datetime] = None,
        end_datetime: Union[int, float, datetime] = None,
        dt: Union[int, float, timedelta, dict] = None,
        rnday: Union[int, float, timedelta] = None,
        filewave: Union[str, os.PathLike] = None,
        proc: Union[f90nml.namelist.Namelist, dict] = None,
        coupl: Union[f90nml.namelist.Namelist, dict] = None,
        grid: Union[f90nml.namelist.Namelist, dict] = None,
        init: Union[f90nml.namelist.Namelist, dict] = None,
        bouc: Union[f90nml.namelist.Namelist, dict] = None,
        wind: Union[f90nml.namelist.Namelist, dict] = None,
        curr: Union[f90nml.namelist.Namelist, dict] = None,
        walv: Union[f90nml.namelist.Namelist, dict] = None,
        engs: Union[f90nml.namelist.Namelist, dict] = None,
        nums: Union[f90nml.namelist.Namelist, dict] = None,
        history: Union[f90nml.namelist.Namelist, dict] = None,
        station: Union[f90nml.namelist.Namelist, dict] = None,
        hotfile: Union[f90nml.namelist.Namelist, dict] = None,
        petscoptions: Union[f90nml.namelist.Namelist, dict] = None,
        nesting: Union[f90nml.namelist.Namelist, dict] = None,
        **kwargs
        # schism_param: TO DO -- make param.nml mirror param setup for schism
    ):
        """
        Initialize WWM_Param from a template and optional user-defined section options.

        Default values need to be changed! 
        """
        # Define all section names
        self.section_names = [
            "proc", 
            "coupl", 
            "grid", 
            "init", 
            "bouc", 
            "wind", 
            "curr",
            "walv", 
            "engs", 
            "nums", 
            "history", 
            "station",
            "hotfile",
            "petscoptions", 
            "nesting"
        ]

        # Load the namelist from the template
        if isinstance(template, f90nml.namelist.Namelist):
            nml = template
        else:
            parser = f90nml.Parser()
            nml = parser.read(template)
        self.nml = nml

        # Get provided section arguments (if any) from locals
        provided_args = {
            key: value for key, value in locals().items()
            if key in self.section_names and value is not None
        }

        # Patch the main namelist for each provided section
        for section, value in provided_args.items():
            if not isinstance(value, f90nml.namelist.Namelist):
                value = f90nml.namelist.Namelist(value)
            self.nml.patch(value)

        # clear unused or out-of-date sections
        self.nml.patch({'station':{}})
        self.nml.patch({'wind':{}})
        self.nml.patch({'curr':{}})
        self.nml.patch({'walv':{}})


        # TO DO: 
        #  - consider create an attribute or class for each section (like pyschism.param? 
        #  - implement logic for connected variables in namelist and WWM ... very convoluted at this point
        #
        # for section in self.section_names:
        #     section_data = self.nml[section].todict() if section in self.nml else {}
        #     setattr(self, section, section_data)

        self.update_datetime(start_datetime=start_datetime, end_datetime=end_datetime, dt=dt, rnday=rnday)
        
        if filewave is not None:
            self.update_bouc_datetime(filewave=filewave)

    def update_datetime(self, start_datetime: Union[int, float, datetime], end_datetime: Union[int, float, datetime], dt: Union[int, float, timedelta, dict], rnday: Union[int, float, timedelta]=None, sections: list = None):
        """
        Handle time inputs in namelist sections

            start_datetime : start datetime of WWM simulation. Same values for all sections 'proc', 'history', 'station', 'hotfile'
            end_datetime : end datetime of WWM simulation. Same value for all sections 'proc', 'history', 'station', 'hotfile'
            rnday : duration of simulation ... must agree with start_datetime and end_datetime.
            dt: dict like {'proc':timedelta(seconds=6000),'history':timedelta(seconds=3600)} or use default values. 

        Note! 'bouc' section time variables are not modified here. 

        TO DO: have a more dynamic way to set deltc (dt) values. It can be passed in as a 
        """
        # If no sections are given, update all sections with defaults
        if sections is None:
            sections = ['proc', 'history', 'station', 'hotfile']

        def format_datetime(datetime_value):
            """
            Convert a datetime object to a formatted string 'yyyymmdd.hhmmss'.
            If dt_value is not a datetime, return it unchanged.
            """
            if isinstance(datetime_value, datetime):
                return datetime_value.strftime("%Y%m%d.%H%M%S")
            return datetime_value

        def convert_to_seconds(x):
            """
            Convert a timedelta to seconds.
            If x is already an int or float, return it as a float.
            """
            if isinstance(x, (int, float)):
                return float(x)
            elif isinstance(x, timedelta):
                return x.total_seconds()
            else:
                raise TypeError("Unsupported type for conversion to seconds.")

        if start_datetime is not None:
            self.start_datetime = start_datetime
        else:
            start_datetime = self.nml['proc']['begtc']
        if end_datetime is not None:     
            self.end_datetime = end_datetime
        if dt is not None:
            self.dt = dt
        if rnday is not None:
            self.rnday = rnday

        # Validate and/or compute the time-related variables
        if start_datetime is not None and end_datetime is not None and rnday is not None:
            # Compute the difference between end and start in seconds
            if isinstance(start_datetime, datetime) and isinstance(end_datetime, datetime):
                diff = (end_datetime - start_datetime).total_seconds()
            else:
                diff = float(end_datetime) - float(start_datetime)
            rnday_seconds = convert_to_seconds(rnday)
            if rnday_seconds != diff:
                raise ValueError("runday must equal end_datetime - start_datetime")
        elif start_datetime is not None and end_datetime is not None:
            # Compute rnday from the difference between end and start
            if isinstance(start_datetime, datetime) and isinstance(end_datetime, datetime):
                rnday = end_datetime - start_datetime  # This will be a timedelta
            else:
                rnday = float(end_datetime) - float(start_datetime)
        elif start_datetime is not None and rnday is not None and end_datetime is None:
            # Compute end_datetime using start_datetime and rnday
            if isinstance(start_datetime, datetime):
                if isinstance(rnday, timedelta):
                    end_datetime = start_datetime + rnday
                else:
                    end_datetime = start_datetime + timedelta(seconds=rnday)
            else:
                end_datetime = float(start_datetime) + convert_to_seconds(rnday)

        # Loop over each section and patch the namelist
        for section in sections:
            if isinstance(dt, dict):
                dt_val = dt[section]
                dt_val = dt_val.total_seconds() if isinstance(dt_val, timedelta) else dt_val
            else:
                if section in ('proc', 'bouc'):
                    # For 'proc', use the provided dt (convert if necessary)
                    dt_val = dt.total_seconds() if isinstance(dt, timedelta) else dt
                elif section in ('history', 'station'):
                    dt_val = 3600.0  # seconds (1 hour)
                elif section == 'hot':
                    dt_val = 86400.0  # seconds (1 day)

            # Format the start and end datetime values for the namelist.
            begtc = format_datetime(start_datetime)
            endtc = format_datetime(end_datetime)

            # Create the patch value for this section
            val = {
                section: {
                    'begtc': begtc,  # Expected format: yyyymmdd.hhmmssg
                    'deltc': dt_val,  # dt in seconds
                    'unitc': 'sec',
                    'endtc': endtc   # Expected format: yyyymmdd.hhmmss
                }
            }

            print(f"Patching {section} time variables: {val}")
            self.nml.patch(val)
            
        return self
        
    def update_bouc_datetime(self, filewave: Union[str, os.PathLike]):
        """
        Set time variables in the 'bouc' section of the namelist based on filewave.

        filewave : data file or ASCII file with filenames of .nc files used to force WWM.
                - If filewave contains timeseries data, it is assumed that the first column of
                    each line is a numeric time value (epoch seconds).
                - If filewave contains netCDF filenames, each line is assumed to be a netCDF file,
                    and the time variable (assumed to be named 'time') is parsed from each file.

        This method computes and patches the following variables in the 'bouc' section:
            BEGTC  : Begin time, formatted as 'yyyymmdd.hhmmss'
            DELTC  : Time step (in seconds)
            UNITC  : Unit of time as 'sec' 'min' 'hr'
            ENDTC  : End time, formatted as 'yyyymmdd.hhmmss'
        """
        import os
        from datetime import datetime
        import netCDF4

        # Read non-empty lines from the filewave file.
        with open(filewave, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Determine the file type:
        first_token = lines[0].split()[0]
        try:
            float(first_token) # If the first token of the first line is numeric, assume timeseries data.
            mode = 'timeseries'
        except ValueError:
            mode = 'netcdf'


        if mode == 'timeseries':
            raise ValueError('.update_bouc_datetime method not implemented')
        
        if mode == 'netcdf':
        
            # Initialize lists to store times from all files
            start_times = []
            end_times = []
            dt_values = []

            for fname in lines:
                if not os.path.isabs(fname):
                    fname = os.path.join(os.path.dirname(filewave), fname)  # Make relative paths absolute
                
                if os.path.exists(fname):
                    try:
                        print(f'Parsing Times: {fname}')
                        ds = xr.open_dataset(fname, decode_times=True)  # do not decode time
                        times = ds['time'].values  
                        ds.close()
                    except Exception as e:
                        print(f"Error reading netCDF file {fname}: {e}")
                        continue  # Skip this file and continue with the others

                    if times.size == 0:
                        raise ValueError(f"No valid time data found in netCDF file: {fname}")

                    # Convert start and end times to Python datetime
                    start_datetime = times[0].astype('M8[ms]').astype(datetime)
                    end_datetime = times[-1].astype('M8[ms]').astype(datetime)

                    # Compute time step in seconds (assuming uniform dt)
                    dt_seconds = (times[1] - times[0]) / np.timedelta64(1, 's')

                    # Store values for comparison
                    start_times.append(start_datetime)
                    end_times.append(end_datetime)
                    dt_values.append(dt_seconds)

            # Check that all files have the same start time, end time, and dt
            if not (all(t == start_times[0] for t in start_times) and
                    all(t == end_times[0] for t in end_times) and
                    all(dt == dt_values[0] for dt in dt_values)):
                raise ValueError("Mismatch in time variables across netCDF files in filewave!")

            # Use the consistent values for updating
            self.update_datetime(
                start_datetime=start_times[0],
                end_datetime=end_times[0],
                dt=dt_values[0],
                sections=['bouc']
            )

        return self


    def disp(self):
        return print(json.dumps(self.nml.todict(), indent=4))

    def write(self, filename: Union[str, os.PathLike] = "wwminput.nml",force=False,sort=False):
        """ 
        Write the updated namelist to a file.
        """
        # Convert to uppercase for readability
        self.nml.uppercase = True

        # Write the updated namelist
        f90nml.write(self.nml, filename,force=force,sort=sort)

        print(f"Namelist written to {filename}")
