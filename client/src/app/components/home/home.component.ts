import { Component } from '@angular/core';
import { ISearchResults } from '../../entities/search-results.interface.js';
import { SearchService } from '../../services/search.service.js';

@Component({
    selector: 'ss-search',
    templateUrl: 'src/app/components/home/home.component.html',
    styleUrls: ['src/app/components/home/home.component.css']
})

export class HomeComponent {
    searchToken: string = '';
    logo: string = 'src/media/logo.jpg';

    constructor(private _searchService: SearchService){ }
}