#ifndef _LOGGING_H_
#define _LOGGING_H_




#include <iostream>

void print_header(){

  std::cout << "n_devices "
	    << "dev_type "
	    << "dev_name "
	    << "n_repeats "
	    << "total_time_ms "
	    << "stack_dims_x "
	    << "stack_dims_y "
	    << "stack_dims_z "
	    << "type_width_byte "
	    << "comment\n";

}

template <typename T>
void print_info(int		 num_devices			,
		std::string	 dev_type		,
		std::string	 dev_name		,
		int		 num_repeats			,
		double		 total_time_ms		,
		std::vector<T> stack_dims	,
		int type_width_byte,
		std::string	 comment
		){

  std::cout << num_devices    << " "
	    << dev_type       << " "
	    << dev_name       << " "
	    << num_repeats    << " "
	    << total_time_ms  << " "
	    << stack_dims[0]     << " "
	    << stack_dims[1]     << " "
	    << stack_dims[2]     << " "
	    << comment        << "\n";

}

#endif /* _LOGGING_H_ */
