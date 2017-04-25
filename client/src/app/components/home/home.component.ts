import { Component } from '@angular/core';
import { ISearchResults } from '../../entities/search-results.interface.js';
import { SearchService } from '../../services/search.service.js';
import { SharedService } from '../../services/shared.service.js';

@Component({
    selector: 'ss-search',
    templateUrl: 'src/app/components/home/home.component.html',
    styleUrls: ['src/app/components/home/home.component.css']
})

export class HomeComponent {
    searchToken: string = '';
    logo: string = 'src/media/logo.jpg';

    constructor(private _searchService: SearchService, private _sharedService: SharedService){
    }

    search() {
        this._searchService.search(this.searchToken)
            .map(response => <ISearchResults>response)
            .subscribe( 
                results => { this._sharedService.addSearchResults(results); 
                             this._sharedService.addSearchTerm(this.searchToken) },
                error => console.log(error)
            );
    }
}